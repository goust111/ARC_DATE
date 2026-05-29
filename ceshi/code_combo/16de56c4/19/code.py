# Combo variant of 16de56c4 (index 19)
# combo_spec: D12/2+D13/2
# Name: 反射传播+L形路径
#
# Design analysis
# a) Concrete behavior and trace
# - The source task is a one-dimensional beat fill: the repeated marker pair reveals
#   a step, and an aligned companion only truncates / recolors that same row or column.
# - In this combo the repeated pair still exposes the step and the unique shooting edge
#   still chooses the initial axis, but aligned singleton companions now behave like
#   one-use mirrors encountered in sequence.
# - Small left-edge example, step=2 on row r:
#   marker 1 at (r,0) and (r,2), companion 6 at (r,6), companion 7 at (r+4,6).
#   Original logic would stay on row r only.  The combo does:
#     (r,0)->(r,2)->(r,4) painted 1,
#     hit (r,6): paint it 6 and turn downward,
#     (r+2,6) painted 6,
#     hit (r+4,6): paint it 7 and turn left,
#     (r+4,4)->(r+4,2)->(r+4,0) painted 7, then exit.
#   One companion yields an L; two companions yield a sparse Z-like path.
#
# b) Complexity calibration
# - Every active source line has at least one companion, so samples cannot collapse
#   back to the original straight 1-D propagation.
# - The generator uses only 1-turn and 2-turn rays and enforces at least one 2-turn
#   line per grid.  That keeps the rule visible while avoiding an unreadable maze.
# - Colors change only at visible corner dots.  No color ordering or numeric trick is
#   used; the geometry itself explains why the new segment changes color.
#
# c) Hidden assumptions from the original solver that must change
# - "All relevant cells lie on one source row / column" is false after the first turn.
#   The solver has to walk a 2-D path instead of looping over one line only.
# - "The first aligned companion fully determines the affected span" is false.  There
#   can now be an ordered sequence of companions on alternating axes.
# - "A companion recolors an entire source-line span in one shot" is false.  The new
#   process is sequential: paint until a corner, adopt that corner color, rotate 90
#   degrees, then continue with the same step on the orthogonal axis.
#
# d) Placement legality guaranteed by the generator
# - Each ray still starts from exactly one shooting-edge line with a repeated marker
#   color appearing twice: one boundary marker and one inward marker exactly one step in.
# - Companions are interior singleton cells placed on later beat positions of the same
#   ray; they never touch non-shooting boundaries and never land on another active
#   source line, so row/column parsing stays unambiguous.
# - Full output paths are built output-first and rejected on any input/output overlap,
#   so no ray crosses another ray or runs through a foreign input dot.
#
# e) Visual observability
# - The changed behavior is visible from shape alone: outputs contain periodic elbows
#   and Z paths rather than straight arithmetic progressions.
# - Because every line turns at least once and at least one line turns twice, a solver
#   looking at only 2-4 examples can infer both the right-angle rule and the fact that
#   turning can repeat.
#
# f) Visual cleanliness
# - Paths are sparse, non-overlapping, and use only the original step spacing.
# - The generator forbids stacked corner dots on the same active source lines, which
#   keeps each ray attributable to a single launcher.
#
# g) Color inference
# - The repeated marker color only identifies the launching beat.  Each companion dot
#   donates its own color to the corner cell and to every later cell on that segment.
# - The number of distinct colors stays moderate because colors are reused across rays;
#   the puzzle never relies on a per-color lookup table hidden from the viewer.
#
# Post-change assumption audit
# - `main()` assumes every active source line is identified by a non-bg dot on the
#   unique shooting edge.  The generator guarantees exactly one such marker per active
#   line and keeps all companions off non-shooting boundaries.
# - `main()` assumes any later non-bg cell reached by a ray belongs to that same ray and
#   should trigger a turn.  The generator guarantees path non-overlap and forbids input
#   dots on another ray's path.
# - `main()` assumes repeated colors occur only for the launching marker pair on each
#   source line.  The generator uses singleton companions and avoids placing them on
#   foreign active lines, so line parsing remains valid.

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


def _shooting_edge(g: np.ndarray, bg: int) -> str:
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
    return winners[0]


def _turn_vector(edge: str, dr: int, dc: int) -> tuple[int, int]:
    if edge in ("left", "right"):
        return int(dc), int(-dr)      # clockwise turn
    return int(-dc), int(dr)          # counter-clockwise turn


def _initial_state(edge: str, line: int, H: int, W: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return {
        "left": ((line, 0), (0, 1)),
        "right": ((line, W - 1), (0, -1)),
        "top": ((0, line), (1, 0)),
        "bottom": ((H - 1, line), (-1, 0)),
    }[edge]


def _source_lines(g: np.ndarray, edge: str, bg: int) -> list[int]:
    H, W = g.shape
    if edge == "left":
        return [r for r in range(H) if g[r, 0] != bg]
    if edge == "right":
        return [r for r in range(H) if g[r, W - 1] != bg]
    if edge == "top":
        return [c for c in range(W) if g[0, c] != bg]
    return [c for c in range(W) if g[H - 1, c] != bg]


def _line_profile(
    g: np.ndarray,
    line: int,
    *,
    edge: str,
    bg: int,
) -> dict:
    by_row = edge in ("left", "right")
    line_len = g.shape[1] if by_row else g.shape[0]

    def cell(pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    colors: dict[int, list[int]] = defaultdict(list)
    for p in range(line_len):
        value = int(g[cell(p)])
        if value != bg:
            colors[value].append(p)

    repeating = {color: ps for color, ps in colors.items() if len(ps) >= 2}
    assert len(repeating) == 1, (
        f"Line {line} ({'row' if by_row else 'col'}) needs exactly one repeating "
        f"marker color: {dict(colors)}"
    )
    marker, marker_positions = next(iter(repeating.items()))
    assert len(marker_positions) == 2, (
        f"Line {line} ({'row' if by_row else 'col'}) marker {marker} has "
        f"{len(marker_positions)} dots (expected 2): {marker_positions}"
    )

    edge_pos = 0 if edge in ("left", "top") else line_len - 1
    assert edge_pos in marker_positions, (
        f"Line {line} ({'row' if by_row else 'col'}) marker {marker} must touch "
        f"the shooting edge at {edge_pos}: {marker_positions}"
    )

    a, b = sorted(marker_positions)
    step = b - a
    assert step >= 1
    if edge in ("left", "top"):
        assert marker_positions == [0, step], (
            f"Line {line} marker pair must start at the shooting edge and continue "
            f"one beat inward: {marker_positions}"
        )
    else:
        assert marker_positions == [line_len - 1 - step, line_len - 1], (
            f"Line {line} marker pair must end at the shooting edge and continue "
            f"one beat inward: {marker_positions}"
        )

    marker_cells = {cell(pos) for pos in marker_positions}
    return {
        "marker": int(marker),
        "step": int(step),
        "marker_cells": marker_cells,
    }


def _walk_ray(
    g: np.ndarray,
    profile: dict,
    *,
    edge: str,
    line: int,
    bg: int,
    out: np.ndarray | None = None,
) -> int:
    H, W = g.shape
    marker = int(profile["marker"])
    step = int(profile["step"])
    marker_cells: set[tuple[int, int]] = set(profile["marker_cells"])

    pos, (dr, dc) = _initial_state(edge, line, H, W)
    cur_color = marker
    turns = 0

    for _ in range(H * W + 5):
        nr = int(pos[0] + dr * step)
        nc = int(pos[1] + dc * step)
        if not (0 <= nr < H and 0 <= nc < W):
            return turns

        pos = (nr, nc)
        original = int(g[pos])
        if original != bg and pos not in marker_cells:
            turns += 1
            cur_color = original
            if out is not None:
                out[pos] = cur_color
            dr, dc = _turn_vector(edge, dr, dc)
            continue

        if out is not None:
            cur = int(out[pos])
            if cur == bg or pos in marker_cells:
                out[pos] = cur_color

    raise AssertionError(f"Ray on line {line} did not terminate: {profile}")


def _combo_signature(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    edge = _shooting_edge(g, bg)
    lines = _source_lines(g, edge, bg)

    total_turns = 0
    two_turn_lines = 0
    one_turn_lines = 0
    for line in lines:
        profile = _line_profile(g, line, edge=edge, bg=bg)
        turns = _walk_ray(g, profile, edge=edge, line=line, bg=bg, out=None)
        total_turns += turns
        if turns >= 2:
            two_turn_lines += 1
        if turns >= 1:
            one_turn_lines += 1

    return {
        "active_lines": len(lines),
        "one_turn_lines": one_turn_lines,
        "two_turn_lines": two_turn_lines,
        "total_turns": total_turns,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace periodic rays from one dominant edge, turning at aligned corner dots.

    Input structure:
    - The background is the majority color, and sparse colored dots sit on a subset
      of rows or columns determined by one unique shooting edge.
    - Each active source line contains exactly one repeated marker color with two
      dots: one on the shooting edge and one one beat inward, so their gap reveals
      the step size.
    - Additional singleton dots are companions placed on later beat cells of the
      same ray.  They are visible corners where the ray changes direction and may
      adopt a new color.

    Rule:
    - Find the unique shooting edge.  Left/right edges launch along rows;
      top/bottom edges launch along columns.
    - For each active source line, launch a ray from the boundary marker and move
      by the inferred step.  Marker-colored cells stay marker-colored until the ray
      reaches a companion.
    - Landing on a companion repaints that corner with the companion color and turns
      the ray 90 degrees.  The turn handedness is fixed by the edge: left/right use
      clockwise turns; top/bottom use counter-clockwise turns.
    - After each turn, continue with the same step on the new axis until another
      companion is reached or the ray exits the grid.  Earlier segments keep their
      old colors; only the current and later segments use the newest corner color.

    Key insight: the repeated pair still tells you the beat, but the task is no
    longer a one-dimensional span fill.  Aligned singleton dots must be read as an
    ordered sequence of elbow mirrors, so solving requires following one periodic
    path through several right-angle turns.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    edge = _shooting_edge(g, bg)

    for line in _source_lines(g, edge, bg):
        profile = _line_profile(g, line, edge=edge, bg=bg)
        _walk_ray(g, profile, edge=edge, line=line, bg=bg, out=out)

    return out


_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 6)
_STEP_RANGE = (1, 5)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")
_MAX_RETRIES = 400
_EDGE_CYCLE = 0


def _spaced_indices(n: int, total: int) -> list[int]:
    available = total - (n - 1)
    if available < n:
        return []
    picks = sorted(random.sample(range(available), n))
    return [picks[i] + i for i in range(n)]


def _allowed_lines(edge: str, H: int, W: int) -> list[int]:
    if edge == "left":
        return list(range(0, H - 1))
    if edge == "right":
        return list(range(1, H))
    if edge == "top":
        return list(range(0, W - 1))
    return list(range(1, W))


def _orth_room(edge: str, line: int, H: int, W: int) -> int:
    if edge == "left":
        return H - 1 - line
    if edge == "right":
        return line
    if edge == "top":
        return W - 1 - line
    return line


def _line_order(edge: str, lines: list[int], H: int, W: int) -> list[int]:
    return sorted(lines, key=lambda line: _orth_room(edge, line, H, W))


def _choose_dims_and_lines(edge: str) -> tuple[int, int, list[int]] | None:
    by_row = edge in ("left", "right")
    if by_row:
        H = random.randint(max(_ROWS_RANGE[0], 8), _ROWS_RANGE[1])
        W = random.randint(*_COLS_RANGE)
    else:
        H = random.randint(*_ROWS_RANGE)
        W = random.randint(max(_COLS_RANGE[0], 10), _COLS_RANGE[1])

    allowed = _allowed_lines(edge, H, W)
    max_active = min(_ACTIVE_LINES_RANGE[1], (len(allowed) + 1) // 2)
    if max_active < _ACTIVE_LINES_RANGE[0]:
        return None
    n_active = random.randint(_ACTIVE_LINES_RANGE[0], max_active)
    picks = _spaced_indices(n_active, len(allowed))
    if not picks:
        return None
    active_lines = [allowed[i] for i in picks]
    return H, W, _line_order(edge, active_lines, H, W)


def _cells_until_exit(
    start: tuple[int, int],
    direction: tuple[int, int],
    step: int,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    while True:
        r += dr * step
        c += dc * step
        if not (0 <= r < H and 0 <= c < W):
            return cells
        cells.append((int(r), int(c)))


def _on_any_boundary(r: int, c: int, H: int, W: int) -> bool:
    return r == 0 or r == H - 1 or c == 0 or c == W - 1


def _hits_other_active_line(
    cell: tuple[int, int],
    *,
    by_row: bool,
    source_line: int,
    active_lines: set[int],
) -> bool:
    axis = cell[0] if by_row else cell[1]
    return axis in active_lines and axis != source_line


def _build_output_cells(
    start: tuple[int, int],
    direction: tuple[int, int],
    step: int,
    companion_cells: list[tuple[int, int]],
    companion_colors: list[int],
    marker_color: int,
    edge: str,
    H: int,
    W: int,
) -> list[tuple[int, int, int]]:
    out_cells = [(int(start[0]), int(start[1]), int(marker_color))]
    pos = start
    dr, dc = direction
    cur_color = int(marker_color)

    for idx, companion in enumerate(companion_cells):
        while True:
            pos = (int(pos[0] + dr * step), int(pos[1] + dc * step))
            assert 0 <= pos[0] < H and 0 <= pos[1] < W
            if pos == companion:
                cur_color = int(companion_colors[idx])
            out_cells.append((int(pos[0]), int(pos[1]), int(cur_color)))
            if pos == companion:
                dr, dc = _turn_vector(edge, dr, dc)
                break

    while True:
        nr = int(pos[0] + dr * step)
        nc = int(pos[1] + dc * step)
        if not (0 <= nr < H and 0 <= nc < W):
            return out_cells
        pos = (nr, nc)
        out_cells.append((nr, nc, int(cur_color)))


def _build_candidate(
    edge: str,
    line: int,
    H: int,
    W: int,
    n_turns: int,
    active_lines: set[int],
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]] | None:
    by_row = edge in ("left", "right")
    start, direction = _initial_state(edge, line, H, W)
    steps = list(range(_STEP_RANGE[0], _STEP_RANGE[1] + 1))
    random.shuffle(steps)

    for step in steps:
        first_leg = _cells_until_exit(start, direction, step, H, W)
        if len(first_leg) < 3:
            continue

        first_turn_options = [
            cell
            for cell in first_leg[1:-1]
            if not _on_any_boundary(cell[0], cell[1], H, W)
        ]
        random.shuffle(first_turn_options)

        for comp1 in first_turn_options:
            turned_dir = _turn_vector(edge, *direction)
            second_leg = _cells_until_exit(comp1, turned_dir, step, H, W)
            if n_turns == 1:
                if len(second_leg) < 1:
                    continue
                marker_color = random.choice(_COLORS)
                comp1_color = random.choice([c for c in _COLORS if c != marker_color])
                input_cells = [
                    (int(start[0]), int(start[1]), int(marker_color)),
                    (int(first_leg[0][0]), int(first_leg[0][1]), int(marker_color)),
                    (int(comp1[0]), int(comp1[1]), int(comp1_color)),
                ]
                output_cells = _build_output_cells(
                    start,
                    direction,
                    step,
                    [comp1],
                    [comp1_color],
                    marker_color,
                    edge,
                    H,
                    W,
                )
                return input_cells, output_cells

            if len(second_leg) < 3:
                continue
            second_turn_options = [
                cell
                for cell in second_leg[1:-1]
                if not _on_any_boundary(cell[0], cell[1], H, W)
                and not _hits_other_active_line(
                    cell,
                    by_row=by_row,
                    source_line=line,
                    active_lines=active_lines,
                )
            ]
            random.shuffle(second_turn_options)

            for comp2 in second_turn_options:
                third_dir = _turn_vector(edge, *turned_dir)
                third_leg = _cells_until_exit(comp2, third_dir, step, H, W)
                if len(third_leg) < 1:
                    continue

                marker_color = random.choice(_COLORS)
                free_colors = [c for c in _COLORS if c != marker_color]
                comp1_color, comp2_color = random.sample(free_colors, 2)
                input_cells = [
                    (int(start[0]), int(start[1]), int(marker_color)),
                    (int(first_leg[0][0]), int(first_leg[0][1]), int(marker_color)),
                    (int(comp1[0]), int(comp1[1]), int(comp1_color)),
                    (int(comp2[0]), int(comp2[1]), int(comp2_color)),
                ]
                output_cells = _build_output_cells(
                    start,
                    direction,
                    step,
                    [comp1, comp2],
                    [comp1_color, comp2_color],
                    marker_color,
                    edge,
                    H,
                    W,
                )
                return input_cells, output_cells

    return None


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
    for other in _EDGES:
        if other == edge:
            continue
        check = boundary[other]
        counts[other] = sum(1 for r, c in dots_rc if check(r, c))
    return counts


def _gen_attempt(edge: str | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    if edge is None:
        edge = random.choice(_EDGES)
    dims = _choose_dims_and_lines(edge)
    if dims is None:
        return None
    H, W, active_lines = dims
    active_set = set(active_lines)

    inp = np.zeros((H, W), dtype=int)
    expected = np.zeros((H, W), dtype=int)
    occupied_input: set[tuple[int, int]] = set()
    occupied_output: set[tuple[int, int]] = set()

    planned_turns = [2]
    while len(planned_turns) < len(active_lines):
        planned_turns.append(random.choices([1, 2], weights=[3, 2])[0])
    random.shuffle(planned_turns)

    for line, n_turns in zip(active_lines, planned_turns):
        placed = False
        for _ in range(200):
            candidate = _build_candidate(edge, line, H, W, n_turns, active_set)
            if candidate is None:
                break
            input_cells, output_cells = candidate

            input_coords = {(r, c) for r, c, _ in input_cells}
            output_coords = {(r, c) for r, c, _ in output_cells}

            if input_coords & occupied_input:
                continue
            if input_coords & occupied_output:
                continue
            if output_coords & occupied_input:
                continue
            if output_coords & occupied_output:
                continue

            for r, c, color in input_cells:
                inp[r, c] = color
                occupied_input.add((r, c))
            for r, c, color in output_cells:
                expected[r, c] = color
                occupied_output.add((r, c))
            placed = True
            break
        if not placed:
            return None

    all_rc = list(occupied_input)
    shooting_edge_dots = len(active_lines)
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

    sig = _combo_signature(inp)
    if sig["one_turn_lines"] != len(active_lines):
        return None
    if sig["two_turn_lines"] == 0:
        return None
    if sig["total_turns"] < len(active_lines) + 1:
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    global _EDGE_CYCLE
    target_edge = _EDGES[_EDGE_CYCLE % len(_EDGES)]
    _EDGE_CYCLE += 1
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt(target_edge)
        if result is not None:
            inp, _ = result
            return inp
    raise RuntimeError(
        f"Failed to generate valid input for edge {target_edge!r} after {_MAX_RETRIES} attempts"
    )
