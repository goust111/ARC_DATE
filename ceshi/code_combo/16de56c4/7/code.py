# Combo variant of 16de56c4 (index 7)
# combo_spec: D2/0+D16/0
# Name: 每行独立方向+半行填充

from __future__ import annotations

from collections import defaultdict
import importlib.util
import random

import numpy as np

from arc_utils import get_bg


# Design analysis:
# a) Concrete behavior and trace:
#    - Keep the original sparse "one repeated marker color per active line"
#      grammar, but remove the single global shooting side.
#    - The puzzle still chooses one global axis at a time for cleanliness:
#      either active rows or active columns. Within that axis, each non-empty
#      line chooses its own source side independently.
#    - Example on a row of length 13:
#      left-source half-line: marker color 4 at [0, 2, 8] -> step 2 ->
#      no companion -> fill [0, 2, 4, 6, 8].
#      right-source half-line: marker color 7 at [12, 10, 4] -> step 2 ->
#      no companion -> fill [12, 10, 8, 6, 4].
#      Under the original task those lines would have shared one global side,
#      and without a companion they would have filled the whole row on that
#      beat class. Here the side is local and the default range is finite.
#    - Companion lines still exist. Example: [0, 3] as markers and a color-6
#      companion at 9 gives fill [0, 3, 6, 9] recolored to 6.
# b) Complexity calibration:
#    - Only two side choices exist per active line, and the whole grid still
#      uses a single axis, so the puzzle stays readable.
#    - To make the "half-line" change visible, no-companion lines use a third
#      same-color marker farther inside; otherwise the bounded fill would often
#      add too little visual evidence.
#    - Generated samples mix both opposite sides and include both companion and
#      no-companion lines, so both dimensions of the combo are observable.
# c) Original assumptions that break:
#    - `edge_counts` can no longer determine the source side, only a baseline
#      axis-agnostic old solver would try that and fail on mixed-side grids.
#    - A line is no longer required to agree with one global border; instead it
#      must expose its own side by touching exactly one border of the chosen axis.
#    - The old no-companion assumption "fill the entire line" is invalid; now
#      the ray stops at the furthest visible same-color marker.
# d) Placement legality:
#    - Active lines are separated by at least one blank line.
#    - Marker dots on a line are distinct, in-bounds, and lie on one beat class.
#    - Half-lines keep their furthest marker away from the opposite border so
#      the bounded default does not collapse into the original full-line effect.
#    - Bystanders are placed off-beat and away from the borders used for side
#      inference, so they cannot masquerade as companions or source markers.
# e) Visual observability:
#    - Mixed left/right or top/bottom lines make local direction visible in the
#      output, because finite fills grow inward from different sides.
#    - Companion lines provide a second branch with clear recoloring, while
#      half-lines provide the finite-range contrast against the original task.
# f) Visual cleanliness:
#    - The grid remains sparse, active lines are spaced apart, and each line has
#      one repeating color plus at most two singletons, keeping parsing local.
# g) Color inference:
#    - No color-id arithmetic is used. Repeated color means "marker family";
#      aligned foreign color means "absorbing companion"; off-beat foreign dots
#      remain bystanders.
# h) Fusion-specific anti-degeneration checks:
#    - Generator requires both opposite source sides to appear.
#    - Generator requires at least one no-companion half-line and one companion.
#    - Generated output must differ from both a "local-direction + full-line"
#      baseline and a "global-edge + half-line" baseline.
# i) New solver assumptions audited against generation:
#    - Solver assumes exactly one repeated color per non-empty active line and
#      that this repeated color touches exactly one relevant border. Generator
#      enforces both.
#    - Solver infers the step from the two edge-nearest markers of that color;
#      generator always places those markers on the intended beat.
#    - Solver picks the axis whose interpretation explains more valid lines;
#      generator validates that the intended axis wins strictly.


_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_COLORS = list(range(1, 10))

_P_HALF_LINE = 0.55
_P_EXTRA_BYSTANDER = 0.30

_MAX_RETRIES = 700
_ORIGINAL_SOLVER = None


def _cell(by_row: bool, line: int, pos: int) -> tuple[int, int]:
    return (line, pos) if by_row else (pos, line)


def _progress_from_edge(edge: str, line_len: int, pos: int) -> int:
    return pos if edge in ("left", "top") else (line_len - 1 - pos)


def _collect_axis_infos(g: np.ndarray, bg: int, by_row: bool) -> list[dict[str, int | str | None]]:
    H, W = g.shape
    num_lines = H if by_row else W
    line_len = W if by_row else H
    axis_name = "row" if by_row else "col"
    low_edge, high_edge = ("left", "right") if by_row else ("top", "bottom")
    infos: list[dict[str, int | str | None]] = []

    for li in range(num_lines):
        colors: dict[int, list[int]] = defaultdict(list)
        for p in range(line_len):
            value = int(g[_cell(by_row, li, p)])
            if value != bg:
                colors[value].append(p)

        if not colors:
            continue

        total_dots = sum(len(ps) for ps in colors.values())
        assert 2 <= total_dots <= 4, (
            f"Line {li} ({axis_name}) has {total_dots} dots; expected 2..4"
        )

        repeating = [(color, sorted(poses)) for color, poses in colors.items() if len(poses) >= 2]
        assert len(repeating) == 1, (
            f"Line {li} ({axis_name}) must have exactly one repeated color: {dict(colors)}"
        )

        marker, marker_positions = repeating[0]
        assert all(len(poses) == 1 for color, poses in colors.items() if color != marker), (
            f"Line {li} ({axis_name}) has extra repeated foreign colors: {dict(colors)}"
        )

        low_touch = 0 in marker_positions
        high_touch = (line_len - 1) in marker_positions
        assert low_touch ^ high_touch, (
            f"Line {li} ({axis_name}) marker {marker} must touch exactly one border: {marker_positions}"
        )

        edge_name = low_edge if low_touch else high_edge
        edge_pos = 0 if low_touch else line_len - 1
        ordered = sorted(
            marker_positions,
            key=lambda p: _progress_from_edge(edge_name, line_len, p),
        )
        assert ordered[0] == edge_pos, (
            f"Line {li} ({axis_name}) marker {marker} does not start at its local edge: {ordered}"
        )
        assert len(ordered) >= 2, (
            f"Line {li} ({axis_name}) marker {marker} needs at least 2 dots: {ordered}"
        )

        step = _progress_from_edge(edge_name, line_len, ordered[1])
        assert 1 <= step <= min(_STEP_RANGE[1], line_len - 1), (
            f"Line {li} ({axis_name}) illegal step from markers {ordered}"
        )
        offset = edge_pos % step

        for p in ordered:
            progress = _progress_from_edge(edge_name, line_len, p)
            assert progress % step == 0, (
                f"Line {li} ({axis_name}) marker {marker} breaks beat {step}: {ordered}"
            )
            assert p % step == offset, (
                f"Line {li} ({axis_name}) marker {marker} breaks offset {offset}: {ordered}"
            )

        companion_color = None
        companion_pos = None
        for other_color, other_positions in colors.items():
            if other_color == marker:
                continue
            aligned = [p for p in other_positions if p % step == offset]
            if not aligned:
                continue
            assert companion_color is None, (
                f"Line {li} ({axis_name}) has multiple aligned companion colors: {dict(colors)}"
            )
            assert len(aligned) == 1, (
                f"Line {li} ({axis_name}) aligned companion must be a singleton: {dict(colors)}"
            )
            assert _progress_from_edge(edge_name, line_len, aligned[0]) > step, (
                f"Line {li} ({axis_name}) companion is not beyond the first inward marker: {dict(colors)}"
            )
            companion_color = other_color
            companion_pos = aligned[0]

        infos.append(
            {
                "line": li,
                "marker": marker,
                "edge_name": edge_name,
                "step": step,
                "offset": offset,
                "far_marker": ordered[-1],
                "companion_color": companion_color,
                "companion_pos": companion_pos,
                "marker_count": len(ordered),
            }
        )

    return infos


def _choose_axis_infos(g: np.ndarray, bg: int) -> tuple[bool, list[dict[str, int | str | None]]]:
    candidates: list[tuple[bool, list[dict[str, int | str | None]]]] = []
    for by_row in (True, False):
        try:
            infos = _collect_axis_infos(g, bg, by_row)
        except AssertionError:
            continue
        if infos:
            candidates.append((by_row, infos))

    assert candidates, "No valid row-wise or column-wise interpretation found"
    if len(candidates) == 1:
        return candidates[0]

    row_infos = next((infos for by_row, infos in candidates if by_row), [])
    col_infos = next((infos for by_row, infos in candidates if not by_row), [])
    assert len(row_infos) != len(col_infos), (
        f"Ambiguous axis: row_count={len(row_infos)}, col_count={len(col_infos)}"
    )
    return (True, row_infos) if len(row_infos) > len(col_infos) else (False, col_infos)


def _apply_infos(
    g: np.ndarray,
    out: np.ndarray,
    bg: int,
    by_row: bool,
    infos: list[dict[str, int | str | None]],
    full_default: bool,
) -> np.ndarray:
    H, W = g.shape
    line_len = W if by_row else H

    for info in infos:
        li = int(info["line"])
        marker = int(info["marker"])
        edge_name = str(info["edge_name"])
        step = int(info["step"])
        offset = int(info["offset"])
        far_marker = int(info["far_marker"])
        companion_color = info["companion_color"]
        companion_pos = info["companion_pos"]
        edge_pos = 0 if edge_name in ("left", "top") else (line_len - 1)

        if companion_color is not None and companion_pos is not None:
            paint = int(companion_color)
            lo, hi = sorted((edge_pos, int(companion_pos)))
        elif full_default:
            paint = marker
            lo, hi = 0, line_len - 1
        else:
            paint = marker
            lo, hi = sorted((edge_pos, far_marker))

        for p in range(lo, hi + 1):
            if p % step != offset:
                continue
            rc = _cell(by_row, li, p)
            cur = int(out[rc])
            if cur == bg or cur == marker:
                out[rc] = paint

    return out


def _dominant_global_edge(g: np.ndarray, bg: int) -> str:
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
    assert len(winners) == 1, f"Ambiguous dominant edge: {winners} (counts={edge_counts})"
    return winners[0]


def _solve_local_full(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    by_row, infos = _choose_axis_infos(g, bg)
    return _apply_infos(g, out, bg, by_row, infos, full_default=True)


def _solve_global_half(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    edge = _dominant_global_edge(g, bg)
    by_row = edge in ("left", "right")
    H, W = g.shape
    num_lines = H if by_row else W
    line_len = W if by_row else H
    edge_pos = 0 if edge in ("left", "top") else line_len - 1
    axis_name = "row" if by_row else "col"

    for li in range(num_lines):
        colors: dict[int, list[int]] = defaultdict(list)
        for p in range(line_len):
            value = int(g[_cell(by_row, li, p)])
            if value != bg:
                colors[value].append(p)

        if not colors:
            continue

        repeating = [(color, sorted(poses)) for color, poses in colors.items() if len(poses) >= 2]
        assert len(repeating) == 1, (
            f"Line {li} ({axis_name}) must have one repeated color under global-half baseline: {dict(colors)}"
        )
        marker, marker_positions = repeating[0]
        ordered = sorted(
            marker_positions,
            key=lambda p: _progress_from_edge(edge, line_len, p),
        )
        assert ordered[0] == edge_pos, (
            f"Line {li} ({axis_name}) marker {marker} does not touch global edge {edge}: {ordered}"
        )
        step = _progress_from_edge(edge, line_len, ordered[1])
        assert step > 0
        offset = edge_pos % step

        for p in ordered:
            progress = _progress_from_edge(edge, line_len, p)
            assert progress % step == 0 and p % step == offset

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
            lo, hi = min(ordered + companion_positions), max(ordered + companion_positions)
        else:
            paint = marker
            lo, hi = sorted((edge_pos, ordered[-1]))

        for p in range(lo, hi + 1):
            if p % step != offset:
                continue
            rc = _cell(by_row, li, p)
            cur = int(out[rc])
            if cur == bg or cur == marker:
                out[rc] = paint

    return out


def _load_original_solver():
    global _ORIGINAL_SOLVER
    if _ORIGINAL_SOLVER is None:
        path = "/home/lixinyu/snap/new/arc2_solver_select/16de56c4.py"
        spec = importlib.util.spec_from_file_location("orig_16de56c4", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        _ORIGINAL_SOLVER = module
    return _ORIGINAL_SOLVER


def _original_solver_differs(inp: np.ndarray, out: np.ndarray) -> bool:
    module = _load_original_solver()
    try:
        old_out = module.main(inp)
    except Exception:
        return True
    return not np.array_equal(old_out, out)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill locally directed beat rays, each bounded by its own visible evidence.

    Input structure:
    - Background is the most common color and all non-background cells are sparse.
    - The puzzle still uses one global axis at a time: either active rows or
      active columns carry the repeated markers.
    - Each non-empty active line has exactly one repeated marker color. That
      color touches exactly one border of the active axis, so the line itself
      reveals whether it shoots from left/right or top/bottom.
    - Some no-companion lines show an extra same-color marker farther inward;
      foreign-color singletons may be aligned companions or off-beat bystanders.

    Rule:
    - First decide whether rows or columns form the active axis. Then solve each
      non-empty line independently and infer its source side from the border
      touched by the repeated marker color.
    - Sort that repeated color by distance from its local edge. The first inward
      gap gives the step, and all same-color markers stay on that beat class.
    - If an aligned foreign-color dot exists, it becomes the paint color and the
      ray fills only from the local edge to that companion.
    - If no aligned companion exists, fill only from the local edge to the
      furthest visible marker of the repeated color. Bystanders stay unchanged.

    Key insight:
    - The puzzle is no longer controlled by one global shooting border and no
      longer defaults to edge-to-edge propagation. Each line encodes both its
      own source side and its own stopping range.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    by_row, infos = _choose_axis_infos(g, bg)
    return _apply_infos(g, out, bg, by_row, infos, full_default=False)


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


def _candidate_steps_for_half(line_len: int) -> list[int]:
    upper = min(_STEP_RANGE[1], line_len - 1)
    return [step for step in range(_STEP_RANGE[0], upper + 1) if (line_len - 1) // step >= 4]


def _candidate_steps_for_companion(line_len: int) -> list[int]:
    upper = min(_STEP_RANGE[1], line_len - 1)
    steps = []
    for step in range(_STEP_RANGE[0], upper + 1):
        interior = [
            p
            for p in range(2 * step, line_len)
            if p % step == 0 and p not in (0, line_len - 1)
        ]
        if interior:
            steps.append(step)
    return steps


def _edge_pos(side: str, line_len: int) -> int:
    return 0 if side in ("left", "top") else line_len - 1


def _make_pos(side: str, line_len: int, beat: int, step: int) -> int:
    edge_pos = _edge_pos(side, line_len)
    return edge_pos + beat * step if edge_pos == 0 else edge_pos - beat * step


def _nonbeat_positions(side: str, line_len: int, step: int, forbidden: set[int]) -> list[int]:
    offset = _edge_pos(side, line_len) % step
    return [
        p
        for p in range(line_len)
        if p % step != offset and p not in forbidden and p not in (0, line_len - 1)
    ]


def _build_half_line(side: str, line_len: int, marker_color: int) -> tuple[list[tuple[int, int]], str]:
    step = random.choice(_candidate_steps_for_half(line_len))
    max_beat = (line_len - 1) // step
    far_beat = random.randint(3, max_beat - 1)
    positions = [
        _make_pos(side, line_len, 0, step),
        _make_pos(side, line_len, 1, step),
        _make_pos(side, line_len, far_beat, step),
    ]
    dots = [(pos, marker_color) for pos in positions]

    if random.random() < _P_EXTRA_BYSTANDER:
        candidates = _nonbeat_positions(side, line_len, step, set(positions))
        if candidates:
            other_color = random.choice([c for c in _COLORS if c != marker_color])
            dots.append((random.choice(candidates), other_color))

    return dots, "half"


def _build_companion_line(side: str, line_len: int, marker_color: int) -> tuple[list[tuple[int, int]], str]:
    step = random.choice(_candidate_steps_for_companion(line_len))
    marker_positions = [
        _make_pos(side, line_len, 0, step),
        _make_pos(side, line_len, 1, step),
    ]
    companion_candidates = [
        _make_pos(side, line_len, beat, step)
        for beat in range(2, ((line_len - 1) // step) + 1)
    ]
    companion_candidates = [p for p in companion_candidates if p not in (0, line_len - 1)]

    other_color = random.choice([c for c in _COLORS if c != marker_color])
    companion_pos = random.choice(companion_candidates)
    dots = [(marker_positions[0], marker_color), (marker_positions[1], marker_color), (companion_pos, other_color)]

    if random.random() < _P_EXTRA_BYSTANDER:
        forbidden = set(marker_positions + [companion_pos])
        candidates = _nonbeat_positions(side, line_len, step, forbidden)
        if candidates:
            dots.append((random.choice(candidates), random.choice([c for c in _COLORS if c != marker_color])))

    return dots, "companion"


def _axis_signature(inp: np.ndarray) -> tuple[int, int, list[str], list[str], list[dict[str, int | str | None]], list[dict[str, int | str | None]]]:
    bg = int(get_bg(inp))
    try:
        row_infos = _collect_axis_infos(inp, bg, True)
    except AssertionError:
        row_infos = []
    try:
        col_infos = _collect_axis_infos(inp, bg, False)
    except AssertionError:
        col_infos = []
    row_edges = [str(info["edge_name"]) for info in row_infos]
    col_edges = [str(info["edge_name"]) for info in col_infos]
    return len(row_infos), len(col_infos), row_edges, col_edges, row_infos, col_infos


def _line_kinds_from_infos(infos: list[dict[str, int | str | None]]) -> list[str]:
    kinds = []
    for info in infos:
        if info["companion_color"] is not None:
            kinds.append("companion")
        elif int(info["marker_count"]) >= 3:
            kinds.append("half")
        else:
            kinds.append("short_half")
    return kinds


def _gen_attempt() -> np.ndarray | None:
    by_row = random.choice([True, False])
    n_active = random.randint(_ACTIVE_LINES_RANGE[0], _ACTIVE_LINES_RANGE[1])
    dims = _choose_dims(by_row, n_active)
    if dims is None:
        return None
    H, W, num_lines, line_len = dims

    if not _candidate_steps_for_half(line_len) or not _candidate_steps_for_companion(line_len):
        return None

    active_lines = _spaced_indices(n_active, num_lines)
    if not active_lines:
        return None

    side_a, side_b = ("left", "right") if by_row else ("top", "bottom")
    sides = _mixed_sides(side_a, side_b, n_active)
    marker_colors = random.sample(_COLORS, n_active)

    line_kinds = ["half", "companion"]
    while len(line_kinds) < n_active:
        line_kinds.append("half" if random.random() < _P_HALF_LINE else "companion")
    random.shuffle(line_kinds)

    inp = np.zeros((H, W), dtype=int)
    for li, side, marker_color, kind in zip(active_lines, sides, marker_colors, line_kinds):
        dots, _ = _build_half_line(side, line_len, marker_color) if kind == "half" else _build_companion_line(side, line_len, marker_color)
        occupied: set[int] = set()
        for pos, color in dots:
            if pos in occupied:
                return None
            occupied.add(pos)
            r, c = _cell(by_row, li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color

    row_count, col_count, row_edges, col_edges, row_infos, col_infos = _axis_signature(inp)
    infos = row_infos if by_row else col_infos
    edges = row_edges if by_row else col_edges
    other_count = col_count if by_row else row_count
    if len(infos) != n_active or len(infos) <= other_count:
        return None
    if edges.count(side_a) < 2 or edges.count(side_b) < 2:
        return None

    inferred_kinds = _line_kinds_from_infos(infos)
    if "half" not in inferred_kinds or "companion" not in inferred_kinds:
        return None

    try:
        out = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, out):
        return None

    try:
        local_full_out = _solve_local_full(inp)
    except Exception:
        local_full_out = None
    if local_full_out is None or np.array_equal(local_full_out, out):
        return None

    try:
        global_half_out = _solve_global_half(inp)
    except Exception:
        global_half_out = None
    if global_half_out is not None and np.array_equal(global_half_out, out):
        return None

    if not _original_solver_differs(inp, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
