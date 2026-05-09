# Variant of 16de56c4
# Dimension: D17 — 旁观者点行为 (obstacle)
# Change: 旁观者点(不对齐beat的异色点)在输出中完全保持不变... → 旁观者被擦除

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D17 variant
# a) Concrete behavior / trace:
#    - Each active row/column still contains one marker pair whose spacing defines
#      a beat, plus optionally one extra dot of another color.
#    - If the extra dot is aligned to the beat, it is a companion exactly as in the
#      original task: it changes the paint color and caps the painted span.
#    - If the extra dot is off-beat, it is now an obstacle-like bystander that must
#      disappear in the output instead of remaining visible.
#    - Example on a left-shooting row: marker color 2 at positions 0 and 3 gives
#      step=3 and beat {0,3,6,...}. An extra color-7 dot at position 5 is off-beat.
#      Original output kept the 7 at position 5 while filling beat cells with 2.
#      Variant output fills the same beat cells but erases position 5 to background.
# b) Complexity calibration:
#    - The object vocabulary stays the same: one marker pair plus at most one extra
#      dot per active line. This keeps the change focused on classification rather
#      than adding new object types.
#    - To make D17 visually observable, the generator rejects samples with zero
#      bystanders, and also rejects samples where every active line is a bystander
#      line, so the original companion/no-companion logic still appears alongside
#      the new erase behavior.
# c) Hidden assumptions in the original solver:
#    - `out = g.copy()` used to be harmless because bystanders were meant to survive.
#      In the variant this would silently preserve them unless we explicitly erase
#      every off-beat extra dot.
#    - Companion detection previously only needed the first aligned foreign color.
#      That still works, but bystander handling must classify extra dots by position,
#      not just by color, otherwise an off-beat dot would be missed.
#    - The fill loop only overwrites background or marker cells. That assumption is
#      still valid because companions should remain their own color and bystanders
#      are erased separately.
# d) Placement legality:
#    - Marker dots still occupy edge_pos and the first interior beat.
#    - Companion candidates are only chosen on later beat cells away from the edge.
#    - Bystanders are only chosen from non-beat cells distinct from both marker
#      cells, so no overlap or accidental reclassification occurs.
#    - The generator preserves the unique shooting-edge condition by rejecting any
#      grid where some other boundary ties or exceeds the shooting-edge dot count.
# e) Visual observability:
#    - Each generated grid contains at least one true bystander, so some colored dot
#      visibly vanishes from input to output.
#    - Other lines still show regular beat filling and companion-limited filling,
#      giving the human solver contrast needed to infer which extra dots disappear.
# f) Visual cleanliness:
#    - Active lines remain separated by at least one empty line, so fills do not
#      stack on top of one another.
#    - There is never more than one extra dot per active line, keeping the line-wise
#      classification easy to read.
# g) Color inference:
#    - No hidden color ordering is used. The bystander rule depends only on beat
#      alignment, which is visible from marker spacing.
#    - Companion color propagation remains structurally signaled by alignment and
#      span truncation, not by color ids.
# h) Variant metric used in validation:
#    - Count non-background input cells that turn into background in the output.
#      For this variant, every successful generated sample should have a positive
#      count, proving that at least one bystander was erased.
# i) New solver assumptions and generator guarantees:
#    - The solver assumes each active line has exactly one marker color with exactly
#      two dots and at most one aligned companion position; the generator enforces
#      the original one-extra-dot line structure so this remains true.
#    - The solver also assumes bystanders are off-beat relative to the marker pair;
#      the generator constructs them from the explicit non-beat set.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill periodic progressions that shoot from a unique grid edge, while erasing off-beat bystanders.

    Input structure:
    - bg = most common color. Sparse non-bg dots lie on a small set of rows or columns.
    - One boundary has strictly more non-bg dots than the other three; that unique edge is
      the shooting side, so left/right imply row-wise processing and top/bottom imply column-wise.
    - Every active shooting line has exactly one repeated marker color with exactly two dots;
      their spacing defines the beat for that line.
    - Some lines may also contain one extra dot of a different color, either aligned to that
      beat (a companion) or misaligned (a bystander).

    Rule:
    - Detect the unique shooting edge and process each row/column in that direction.
    - For each active line, derive `step` from the gap between the two marker dots and derive
      the beat by `pos % step == offset`.
    - An aligned extra dot is a companion: it changes the fill color to its own color and limits
      filling to the span covered by the markers and that companion.
    - A misaligned extra dot is a bystander: it does not affect the beat and is erased to the
      background color in the output.
    - If there is no companion, fill the entire line's beat positions with the marker color.

    Key insight: the only difficult distinction is whether the odd-colored extra dot shares the
    marker pair's modular beat. Sharing the beat makes it the absorbing companion; missing the
    beat makes it a removable bystander.
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

            aligned_extras: list[tuple[int, int]] = []
            bystander_positions: list[int] = []
            for other_color, positions in colors.items():
                if other_color == marker:
                    continue
                for pos in positions:
                    if pos % step == offset:
                        aligned_extras.append((other_color, pos))
                    else:
                        bystander_positions.append(pos)

            for pos in bystander_positions:
                out[cell(li, pos)] = bg

            if aligned_extras:
                paint, companion_pos = aligned_extras[0]
                lo = min(a, b, companion_pos)
                hi = max(a, b, companion_pos)
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


# --------------- generator (D17 variant) ---------------

_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_LINE_WEIGHTS = {
    "marker_only": 0.30,
    "companion": 0.30,
    "bystander": 0.40,
}
_MIN_BYSTANDER_LINES = 1

_MAX_RETRIES = 300


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
    """Choose n indices from [0, total) with at least one empty index between neighbors."""
    available = total - (n - 1)
    if available < n:
        return []
    picks = sorted(random.sample(range(available), n))
    return [picks[i] + i for i in range(n)]


def _weighted_choice(options: list[str]) -> str:
    weights = [_LINE_WEIGHTS[option] for option in options]
    return random.choices(options, weights=weights, k=1)[0]


def _build_line_dots(
    edge: str, line_len: int, marker_color: int, step: int
) -> tuple[list[tuple[int, int]], int, str]:
    """Build one active line and report whether it became marker_only/companion/bystander."""
    if edge in ("left", "top"):
        edge_pos, inner_pos = 0, step
    else:
        edge_pos, inner_pos = line_len - 1, line_len - 1 - step
    offset = edge_pos % step

    dots = [(edge_pos, marker_color), (inner_pos, marker_color)]
    other_colors = [color for color in _COLORS if color != marker_color]

    if edge in ("left", "top"):
        companion_positions = list(range(inner_pos + step, line_len, step))
    else:
        companion_positions = list(range(inner_pos - step, -1, -step))
    bystander_positions = [
        pos for pos in range(line_len) if pos % step != offset and pos not in (edge_pos, inner_pos)
    ]

    feasible = ["marker_only"]
    if other_colors and companion_positions:
        feasible.append("companion")
    if other_colors and bystander_positions:
        feasible.append("bystander")

    kind = _weighted_choice(feasible)
    if kind == "companion":
        dots.append((random.choice(companion_positions), random.choice(other_colors)))
    elif kind == "bystander":
        dots.append((random.choice(bystander_positions), random.choice(other_colors)))

    return dots, edge_pos, kind


def _count_other_edge_dots(
    dots_rc: list[tuple[int, int]], edge: str, H: int, W: int
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
    """Single attempt to generate a valid D17 input."""
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
    shooting_edge_dots = 0
    all_rc: list[tuple[int, int]] = []
    kinds: list[str] = []

    for li in active_lines:
        marker_color = random.choice(_COLORS)
        step = min(random.randint(*_STEP_RANGE), line_len - 1)
        dots, edge_pos, kind = _build_line_dots(edge, line_len, marker_color, step)

        for pos, color in dots:
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

        shooting_edge_dots += 1
        kinds.append(kind)

    bystander_lines = sum(kind == "bystander" for kind in kinds)
    if bystander_lines < _MIN_BYSTANDER_LINES or bystander_lines == len(kinds):
        return None

    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(count >= shooting_edge_dots for count in other_counts.values()):
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    erased = int(np.count_nonzero((inp != 0) & (out == 0)))
    if erased < _MIN_BYSTANDER_LINES:
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
