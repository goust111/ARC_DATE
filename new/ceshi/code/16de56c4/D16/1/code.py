# Variant of 16de56c4
# Dimension: D16 — 填充范围计算 (dst)
# Change: 无伴随时填整行(0到line_len-1)；有伴随时填min到max的span... → 伴随点外延

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np

try:
    from arc_utils import get_bg
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[5] / "solvers"))
    from arc_utils import get_bg


# Design analysis for the D16 variant
# ===================================
# a) Concrete behavior and trace
#    Each active row/column still contains exactly one repeating marker color
#    with two dots: one on the shooting edge and one one step inward. A single
#    aligned different-colored dot is still the companion. The old task treated
#    the companion as a hard stop and repainted the whole marker-companion span
#    with the companion color. The variant changes only that destination logic:
#    the ray keeps going past the companion to the far end of the line, and the
#    companion is the point where the fill color switches.
#
#    Example on a left-shooting row of length 10:
#    marker color 2 at positions 0 and 3, companion color 6 at position 6.
#    Old output on beat positions (step=3): [0,3,6] became 6 and 9 stayed bg.
#    New output: positions 0 and 3 stay/fill as 2, while 6 and 9 are 6.
#    So the affected pipeline stage is the fill-range computation inside
#    main(): it must become a directional split instead of a bounded span.
#
# b) Complexity calibration
#    The object grammar remains tight: one repeating color per active line,
#    at most one companion or one bystander, step in 1..5, 4..9 active lines.
#    To keep the D16 change visible, the generator enforces at least one
#    companion line per puzzle and chooses companion positions that leave at
#    least one additional beat beyond the companion, so the "extension"
#    is visible in the output instead of collapsing to the original look.
#
# c) Implicit-assumption audit in main()
#    - Edge detection by boundary counts still holds unchanged.
#    - Repeating-color detection (exactly two marker dots) still holds.
#    - The old code assumed "if companion exists, one paint color on [min,max]".
#      That assumption is invalid now. The fill must depend on the shooting
#      direction and compare each beat position with the companion position.
#    - The old overwrite guard "only bg or marker may be painted" is still safe:
#      bystanders must survive, and the companion dot already has the correct
#      color at the switch point.
#
# d) Placement legality
#    Companion lines choose a step small enough that the beat sequence contains
#    edge marker, inner marker, companion, and at least one farther beat; this
#    guarantees the variant's outward extension stays in bounds and visible.
#    Bystanders are sampled only from non-beat cells, so they never collide with
#    the painted progression. Active line indices remain spaced apart.
#
# e) Visual observability
#    Human-solvable examples need both marker-only/full-line behavior and at
#    least one color-switching line. The generator therefore guarantees at least
#    one companion line and at least one non-companion line in each grid.
#    The rule is visible from geometry alone: same beat, same direction, but
#    color changes at the aligned dot and continues outward.
#
# f) Cleanliness
#    Each line carries only one mechanism, so there is no heavy overlap. The
#    only non-bg cells on a line are the two marker dots plus optional companion
#    or bystander, making the resulting rays easy to read.
#
# g) Color inference
#    No ordering or sorting uses color ids. The companion's visible role is
#    structural: its location on the beat determines where the marker-colored
#    ray changes into the companion color.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend periodic rays from a unique shooting edge, switching color at a companion.

    Input structure:
    - bg is the dominant color. Non-bg cells are sparse dots on several lines.
    - Exactly one grid edge has the most non-bg dots; it defines the shooting
      direction (left/right -> rows, top/bottom -> columns).
    - On each active shooting-direction line, one marker color appears exactly
      twice: once on the shooting edge and once one period inward.
    - A different-colored aligned dot on the same modular beat is a companion;
      an unaligned dot is only a bystander and stays unchanged.

    Rule:
    - Find the unique shooting edge from boundary-dot counts.
    - For each active line, use the two marker dots to recover the period.
    - Without a companion, fill every beat position across the whole line with
      the marker color, as in the original task.
    - With a companion, the ray no longer stops there: beat positions between
      the shooting edge and the companion use the marker color, and the beat at
      the companion and every farther beat toward the opposite edge use the
      companion color.

    Key insight:
    - The repeated marker pair still defines the beat, but the aligned foreign
      dot is no longer a wall. It is a color-switch point on the same ray, so
      solving the variant requires tracking direction as well as periodicity.
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
    forward = edge in ("left", "top")

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

            aligned: list[tuple[int, int]] = []
            for other_color, positions in colors.items():
                if other_color == marker:
                    continue
                for p in positions:
                    if p % step == offset:
                        aligned.append((p, other_color))

            if aligned:
                companion_pos, companion_color = (
                    max(aligned) if forward else min(aligned)
                )
            else:
                companion_pos, companion_color = None, None

            for p in range(line_len):
                if p % step != offset:
                    continue

                if companion_pos is None:
                    paint = marker
                elif forward:
                    paint = marker if p < companion_pos else companion_color
                else:
                    paint = marker if p > companion_pos else companion_color

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

_P_BYSTANDER = 0.40
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


def _choose_line_kinds(n_active: int) -> list[str]:
    companion_count = random.randint(1, n_active - 1)
    kinds = ["companion"] * companion_count
    while len(kinds) < n_active:
        kinds.append("bystander" if random.random() < _P_BYSTANDER else "marker_only")
    random.shuffle(kinds)
    return kinds


def _companion_step_limit(line_len: int) -> int:
    return min(_STEP_RANGE[1], (line_len - 1) // 3)


def _build_line(edge: str, line_len: int, line_kind: str, marker_color: int):
    forward = edge in ("left", "top")

    if forward:
        edge_pos = 0
    else:
        edge_pos = line_len - 1

    if line_kind == "companion":
        step_hi = _companion_step_limit(line_len)
        if step_hi < _STEP_RANGE[0]:
            return None
        step = random.randint(_STEP_RANGE[0], step_hi)
    elif line_kind == "bystander":
        step = random.randint(2, min(_STEP_RANGE[1], line_len - 1))
    else:
        step = random.randint(_STEP_RANGE[0], min(_STEP_RANGE[1], line_len - 1))

    inner_pos = edge_pos + step if forward else edge_pos - step
    dots = [(edge_pos, marker_color), (inner_pos, marker_color)]
    info = {
        "marker_color": marker_color,
        "step": step,
        "companion": None,
    }

    other_colors = [color for color in _COLORS if color != marker_color]
    if line_kind == "companion":
        if forward:
            beats = list(range(inner_pos + step, line_len - step, step))
        else:
            beats = list(range(inner_pos - step, step - 1, -step))
        if not beats or not other_colors:
            return None
        pos = random.choice(beats)
        color = random.choice(other_colors)
        dots.append((pos, color))
        info["companion"] = (pos, color)
    elif line_kind == "bystander":
        offset = edge_pos % step
        non_beat = [p for p in range(line_len) if p % step != offset and p not in (edge_pos, inner_pos)]
        if not non_beat or not other_colors:
            return None
        dots.append((random.choice(non_beat), random.choice(other_colors)))

    return dots, info


def _paint_expected_line(
    grid: np.ndarray,
    edge: str,
    line: int,
    line_len: int,
    marker_color: int,
    step: int,
    companion: tuple[int, int] | None,
):
    by_row = edge in ("left", "right")
    forward = edge in ("left", "top")
    offset = 0 if forward else (line_len - 1) % step

    def cell(pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    for p in range(line_len):
        if p % step != offset:
            continue

        if companion is None:
            paint = marker_color
        else:
            companion_pos, companion_color = companion
            if forward:
                paint = marker_color if p < companion_pos else companion_color
            else:
                paint = marker_color if p > companion_pos else companion_color

        r, c = cell(p)
        cur = int(grid[r, c])
        if cur == 0 or cur == marker_color:
            grid[r, c] = paint


def _count_other_edge_dots(dots_rc: list[tuple[int, int]], edge: str, H: int, W: int) -> dict[str, int]:
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

    line_kinds = _choose_line_kinds(n_active)

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    expected = np.zeros((H, W), dtype=int)
    shooting_edge_dots = 0
    all_rc: list[tuple[int, int]] = []
    companion_lines = 0

    for li, line_kind in zip(active_lines, line_kinds):
        marker_color = random.choice(_COLORS)
        built = _build_line(edge, line_len, line_kind, marker_color)
        if built is None:
            return None
        dots, info = built

        for pos, color in dots:
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            expected[r, c] = color
            all_rc.append((r, c))

        _paint_expected_line(
            expected,
            edge,
            li,
            line_len,
            info["marker_color"],
            info["step"],
            info["companion"],
        )

        if info["companion"] is not None:
            companion_lines += 1
        shooting_edge_dots += 1

    if companion_lines == 0 or companion_lines == n_active:
        return None

    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(count >= shooting_edge_dots for count in other_counts.values()):
        return None

    try:
        out = main(inp)
    except Exception:
        return None
    if not np.array_equal(out, expected):
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
