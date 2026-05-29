# Variant of 16de56c4
# Dimension: D2 — 路径方向确定方式 (path)
# Change: 射击边由四条边上非背景点计数最多的边决定，left/right边对应行方向，top/bottom边对应列方向... → 每行独立方向

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


# Design analysis
# a) Concrete behavior and trace:
#    - The original task used one global shooting edge. Here the puzzle still
#      uses one global axis at a time (rows or columns), but each active line
#      chooses its own side from that axis.
#    - To make the side matter in the output, the two repeated marker dots are
#      moved one cell off the actual beat positions. Example on a row:
#        left-source line with step 3: markers at 1 and 4, fill beats at 0,3,6...
#        right-source line with step 3: markers at 7 and 10, fill beats at 11,8,5...
#      The gap still defines the step, but the source side is now needed to know
#      which neighboring beat class to paint.
#    - This avoids degeneration: the old solver cannot recover the new beat
#      class from the marker pair alone, so the variant is behaviorally distinct.
# b) Complexity calibration:
#    - The axis is still global, the data stays sparse, each active line still
#      has exactly one repeated marker color, and companion/bystander roles stay.
#    - The only structural add-on is that markers are off-beat guide dots so the
#      local source side actually changes the filled cells.
# c) Original assumptions that break:
#    - Global `edge_counts` is no longer meaningful.
#    - A repeated color no longer needs a boundary dot; instead it has a dot one
#      cell from a boundary, which reveals the local shooting side.
#    - The filled modular class is no longer the marker modular class. It is the
#      source-boundary class adjacent to those guide dots.
# d) Placement legality:
#    - Every non-empty active line has exactly 2 marker dots of one color at
#      either `[1, 1+step]` or `[L-2-step, L-2]`, plus optional one singleton.
#    - Companions sit on the source-anchored beat positions beyond the farther
#      marker; bystanders sit off-beat. Distinct active lines do not overlap.
# e) Visual observability:
#    - Each generated puzzle contains lines from both opposite sides, so local
#      direction is visible in the output. The newly filled cells hug different
#      boundary-anchored beat classes on different lines.
# f) Visual cleanliness:
#    - Active lines remain sparse and are separated by blank lines, which keeps
#      mixed left/right or top/bottom cases legible.
# g) Color inference:
#    - No color-id arithmetic is used. Marker color means "free propagation";
#      companion color means "absorbing wall". Bystanders stay unchanged.
# h) Variant metric used during validation:
#    - mixed-side count: both opposite sides must appear among active lines.
#    - old-solver divergence: generated samples are rejected unless the original
#      solver either fails or produces a different output on the same input.
# i) New solver assumptions audited against the generator:
#    - Solver assumes one repeated color of size 2 and at most one singleton on
#      each non-empty active line. Generator enforces that exactly.
#    - Solver assumes markers are one cell away from the source boundary and one
#      step apart. Generator constructs only those patterns.
#    - Solver chooses the axis whose interpretation explains more valid lines.
#      Generator validates that the intended axis wins strictly.


def _cell(by_row: bool, line: int, pos: int) -> tuple[int, int]:
    return (line, pos) if by_row else (pos, line)


def _collect_orientation_infos(
    g: np.ndarray, bg: int, by_row: bool
) -> list[dict[str, int | None]]:
    H, W = g.shape
    num_lines = H if by_row else W
    line_len = W if by_row else H
    axis_name = "row" if by_row else "col"
    low_edge, high_edge = ("left", "right") if by_row else ("top", "bottom")
    infos: list[dict[str, int | None]] = []

    for li in range(num_lines):
        colors: dict[int, list[int]] = defaultdict(list)
        for p in range(line_len):
            v = int(g[_cell(by_row, li, p)])
            if v != bg:
                colors[v].append(p)

        if not colors:
            continue

        total_dots = sum(len(ps) for ps in colors.values())
        assert total_dots in (2, 3), (
            f"Line {li} ({axis_name}) has {total_dots} dots; expected 2 or 3"
        )

        repeating = [(c, ps) for c, ps in colors.items() if len(ps) == 2]
        singles = [(c, ps[0]) for c, ps in colors.items() if len(ps) == 1]
        assert len(repeating) == 1, (
            f"Line {li} ({axis_name}) must have exactly one repeated color: {dict(colors)}"
        )
        assert len(singles) <= 1, (
            f"Line {li} ({axis_name}) has too many singleton colors: {dict(colors)}"
        )
        assert all(len(ps) <= 2 for ps in colors.values()), (
            f"Line {li} ({axis_name}) has a color repeated more than twice: {dict(colors)}"
        )

        marker, mpos = repeating[0]
        a, b = sorted(mpos)
        step = b - a
        assert step >= 2, f"Line {li} ({axis_name}) step must be >= 2: {mpos}"

        if a == 1 and b == 1 + step:
            edge_name = low_edge
            offset = 0
            direction = 1
            far_marker = b
        elif b == line_len - 2 and a == line_len - 2 - step:
            edge_name = high_edge
            offset = (line_len - 1) % step
            direction = -1
            far_marker = a
        else:
            raise AssertionError(
                f"Line {li} ({axis_name}) marker {marker} is not a legal local-side guide: {mpos}"
            )

        companion_color = None
        companion_pos = None
        if singles:
            single_color, single_pos = singles[0]
            aligned = (single_pos % step) == offset
            beyond = (single_pos - far_marker) * direction > 0
            if aligned:
                assert beyond, (
                    f"Line {li} ({axis_name}) has aligned singleton on the wrong side: {dict(colors)}"
                )
                companion_color = single_color
                companion_pos = single_pos

        infos.append(
            {
                "line": li,
                "marker": marker,
                "step": step,
                "offset": offset,
                "direction": direction,
                "companion_color": companion_color,
                "companion_pos": companion_pos,
                "edge_name": edge_name,
            }
        )

    return infos


def _apply_infos(
    g: np.ndarray, out: np.ndarray, bg: int, by_row: bool, infos: list[dict[str, int | None]]
) -> np.ndarray:
    H, W = g.shape
    line_len = W if by_row else H

    for info in infos:
        li = int(info["line"])
        marker = int(info["marker"])
        step = int(info["step"])
        offset = int(info["offset"])
        direction = int(info["direction"])
        companion_color = info["companion_color"]
        companion_pos = info["companion_pos"]

        if companion_color is not None and companion_pos is not None:
            paint = int(companion_color)
            lo, hi = (0, int(companion_pos)) if direction > 0 else (int(companion_pos), line_len - 1)
        else:
            paint = marker
            lo, hi = 0, line_len - 1

        for p in range(lo, hi + 1):
            if p % step != offset:
                continue
            rc = _cell(by_row, li, p)
            cur = int(out[rc])
            if cur == bg or cur == marker:
                out[rc] = paint

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill boundary-anchored beat lines whose source side is inferred per line.

    Input structure:
    - Background is the most common color; non-background cells are sparse dots.
    - The puzzle still uses one global axis at a time: either active rows or
      active columns carry the repeated marker pairs.
    - Each non-empty active line has exactly one repeated marker color with two
      dots placed one cell away from a boundary and one step apart.
    - Different active lines may point to opposite sides of the same axis, so
      one row can be left-sourced while another row is right-sourced.

    Rule:
    - First decide whether rows or columns explain the pattern, then solve each
      non-empty line independently.
    - The gap between the two repeated guide dots is the step. Their placement
      reveals the local source side, and the actual beat positions are the
      boundary-anchored progression adjacent to those guide dots.
    - A singleton on that beat progression and farther along the ray is a
      companion: it stops the fill and its color becomes the paint color.
    - A singleton off that progression is a bystander. Without a companion, the
      marker color fills the whole line on the boundary-anchored beat positions.

    Key insight:
    - The repeated dots no longer lie on the painted progression itself. They
      only encode the period and the source side, so local direction must be
      inferred to recover the correct beat class.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    candidates: list[tuple[bool, list[dict[str, int | None]]]] = []
    for by_row in (True, False):
        try:
            infos = _collect_orientation_infos(g, bg, by_row)
        except AssertionError:
            continue
        if infos:
            candidates.append((by_row, infos))

    assert candidates, "No valid row-wise or column-wise interpretation found"

    if len(candidates) == 1:
        by_row, infos = candidates[0]
    else:
        row_candidate = next(item for item in candidates if item[0] is True)
        col_candidate = next(item for item in candidates if item[0] is False)
        assert len(row_candidate[1]) != len(col_candidate[1]), (
            f"Ambiguous axis: row_count={len(row_candidate[1])}, col_count={len(col_candidate[1])}"
        )
        by_row, infos = row_candidate if len(row_candidate[1]) > len(col_candidate[1]) else col_candidate

    return _apply_infos(g, out, bg, by_row, infos)


# --------------- generator (variant) ---------------

_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (2, 5)
_COLORS = list(range(1, 10))

_P_MARKER_ONLY = 0.35
_P_COMPANION = 0.45

_MAX_RETRIES = 500
_ORIGINAL_SOLVER = None


def _choose_dims(by_row: bool, n_active: int) -> tuple[int, int, int, int] | None:
    min_num_lines = 2 * n_active - 1
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


def _mixed_sides(side_a: str, side_b: str, n: int) -> list[str]:
    count_a = random.randint(2, n - 2)
    sides = [side_a] * count_a + [side_b] * (n - count_a)
    random.shuffle(sides)
    return sides


def _build_line_dots(side: str, line_len: int, marker_color: int, step: int) -> tuple[list[tuple[int, int]], str]:
    if side in ("left", "top"):
        marker_positions = [1, 1 + step]
        beat_offset = 0
        direction = 1
        far_marker = 1 + step
    else:
        marker_positions = [line_len - 2 - step, line_len - 2]
        beat_offset = (line_len - 1) % step
        direction = -1
        far_marker = line_len - 2 - step

    dots = [(marker_positions[0], marker_color), (marker_positions[1], marker_color)]
    line_kind = "marker_only"

    roll = random.random()
    if roll < _P_MARKER_ONLY:
        return dots, line_kind

    other_colors = [c for c in _COLORS if c != marker_color]
    if roll < _P_MARKER_ONLY + _P_COMPANION:
        beats = [
            p
            for p in range(line_len)
            if p % step == beat_offset and (p - far_marker) * direction > 0
        ]
        if other_colors and beats:
            dots.append((random.choice(beats), random.choice(other_colors)))
            line_kind = "companion"
        return dots, line_kind

    non_beat = [
        p
        for p in range(line_len)
        if p % step != beat_offset and p not in marker_positions
    ]
    if other_colors and non_beat:
        dots.append((random.choice(non_beat), random.choice(other_colors)))
        line_kind = "bystander"
    return dots, line_kind


def _axis_counts(inp: np.ndarray) -> tuple[int, int, list[str], list[str]]:
    bg = int(get_bg(inp))

    try:
        row_infos = _collect_orientation_infos(inp, bg, True)
    except AssertionError:
        row_infos = []
    try:
        col_infos = _collect_orientation_infos(inp, bg, False)
    except AssertionError:
        col_infos = []

    row_edges = [str(info["edge_name"]) for info in row_infos]
    col_edges = [str(info["edge_name"]) for info in col_infos]
    return len(row_infos), len(col_infos), row_edges, col_edges


def _choose_active_count(by_row: bool) -> int:
    max_lines = _ROWS_RANGE[1] if by_row else _COLS_RANGE[1]
    max_active = min(_ACTIVE_LINES_RANGE[1], (max_lines + 1) // 2)
    return random.randint(_ACTIVE_LINES_RANGE[0], max_active)


def _old_solver_differs(inp: np.ndarray, out: np.ndarray) -> bool:
    global _ORIGINAL_SOLVER
    if _ORIGINAL_SOLVER is None:
        import importlib.util

        path = "/home/lixinyu/snap/new/arc2_solver_select/16de56c4.py"
        spec = importlib.util.spec_from_file_location("orig_16de56c4", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        _ORIGINAL_SOLVER = mod
    try:
        old_out = _ORIGINAL_SOLVER.main(inp)
    except Exception:
        return True
    return not np.array_equal(old_out, out)


def _gen_attempt() -> np.ndarray | None:
    by_row = random.choice([True, False])
    n_active = _choose_active_count(by_row)
    dims = _choose_dims(by_row, n_active)
    if dims is None:
        return None
    H, W, num_lines, line_len = dims

    max_step = min(_STEP_RANGE[1], line_len - 3)
    if max_step < _STEP_RANGE[0]:
        return None

    active_lines = _spaced_indices(n_active, num_lines)
    if not active_lines:
        return None

    side_a, side_b = ("left", "right") if by_row else ("top", "bottom")
    sides = _mixed_sides(side_a, side_b, n_active)
    marker_colors = random.sample(_COLORS, n_active)

    inp = np.zeros((H, W), dtype=int)
    side_kinds: dict[str, list[str]] = {side_a: [], side_b: []}

    for li, side, marker_color in zip(active_lines, sides, marker_colors):
        step = random.randint(_STEP_RANGE[0], max_step)
        dots, line_kind = _build_line_dots(side, line_len, marker_color, step)

        for pos, color in dots:
            r, c = _cell(by_row, li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color

        side_kinds[side].append(line_kind)

    row_count, col_count, row_edges, col_edges = _axis_counts(inp)
    if by_row:
        if row_count != n_active or row_count <= col_count:
            return None
        if row_edges.count(side_a) < 2 or row_edges.count(side_b) < 2:
            return None
    else:
        if col_count != n_active or col_count <= row_count:
            return None
        if col_edges.count(side_a) < 2 or col_edges.count(side_b) < 2:
            return None

    if not side_kinds[side_a] or not side_kinds[side_b]:
        return None
    if "companion" not in side_kinds[side_a] and "companion" not in side_kinds[side_b]:
        return None

    try:
        out = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, out):
        return None
    if not _old_solver_differs(inp, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
