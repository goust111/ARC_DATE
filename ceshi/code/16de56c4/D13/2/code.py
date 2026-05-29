# Variant of 16de56c4
# Dimension: D13 — 行列交互 (meta)
# Change: 行和列独立处理——射线只沿射击方向（行或列）填充，不存在行列交叉... → L形路径
#
# Design analysis for the D13 variant
# ==================================
# Reference-task note:
# - The prompt requires checking the source task `e87109e9` first.
# - In this workspace its JSON exists, but no readable solver `.py` was found after
#   searching the repository, so the turning baseline below is implemented
#   conservatively from the prompt text only: "hit a trigger, then turn once".
#
# a) Concrete behavior and trace
# - Original behavior on a left-edge row:
#   markers at columns 0 and 3 imply step=3.  If a companion sits at column 9,
#   the old solver paints columns 0,3,6,9 on that same row only.
# - New behavior on the same row:
#   the same 0/3 markers still define step=3 and the same aligned companion at 9
#   still changes the paint color, but now the ray forms an elbow:
#   it paints row cells 0,3,6,9, then keeps going down column 9 at rows
#   r,r+3,r+6,... .  For top/bottom edges the roles of row/column are swapped.
# - Visual consequence:
#   companion lines no longer end as bounded straight segments; they become sparse
#   L-shapes whose corner is exactly the companion dot.
#
# b) Complexity calibration
# - To make D13 visibly dominant, the generator uses companion-bearing lines only.
#   This avoids producing many samples that still look like the original
#   row/column-only task.
# - The number of active lines stays in the original rough band (at least 4), but
#   row-based cases are capped by geometry because turnable rows cannot all sit on
#   the blocked boundary and still keep a blank row between actives.
# - Turn directions are fixed by the shooting edge:
#   left->down, right->up, top->right, bottom->left.
#   This adds exactly one new binary-ish structural choice per edge, instead of
#   introducing extra per-line symbols or color-based instructions.
#
# c) Hidden assumptions in the original solver that no longer hold
# - `lo, hi = min(all_ps), max(all_ps)` used to fully describe the affected area.
#   In the variant it only describes the first leg; a second orthogonal leg must
#   be painted from the companion cell.
# - The old fill loop only needed one 1-D modular class on the shooting line.
#   After the turn, the same step must be replayed on the orthogonal axis with the
#   companion as the new origin.
# - Writing only on the current row/column was previously harmless.  In the
#   variant, the turned leg could cross other structures, so the generator must
#   guarantee those orthogonal cells are otherwise empty.
#
# d) Placement legality guarantees used by the generator
# - Every active line has exactly 2 marker dots plus 1 aligned companion dot.
# - Companion dots are kept off the opposite boundary, so the input still has one
#   unambiguous shooting edge and the elbow is visually interior.
# - Candidate L-shapes are rejected if their output cells overlap another line's
#   output cells or any previously placed input dot.  This prevents accidental
#   blocking on the turned leg.
# - Each turned column/row is unique, which removes stacked vertical/horizontal
#   legs and keeps the scene readable.
#
# e) Visual observability
# - The new rule is visible in a single example because the output contains sparse
#   right-angle paths, not just straight periodic strips.
# - The turn is triggered by a visible companion dot placed on the same beat as
#   the markers, so the corner location is inferable from structure rather than
#   color id tricks.
#
# f) Visual cleanliness
# - Outputs are built from non-overlapping L-shapes only; no dense criss-cross
#   mesh is allowed.
# - The generator rejects placements where a future ray would need to pass through
#   a foreign input dot, avoiding ambiguous "why did this pixel stop?" artifacts.
#
# g) Color inference
# - Colors are still local and structural: the repeated color identifies the beat,
#   and the companion color becomes the entire turned path color.
# - No ordering or transformation depends on numeric color ids.
#
# Post-change assumption audit
# - `main()` now assumes that if a companion exists, the orthogonal cells reached
#   by stepping from the companion are background (except the companion itself).
#   The generator enforces this with explicit collision checks.
# - `main()` also assumes the turn direction depends only on the unique shooting
#   edge, not on the marker/companion color.  The generator varies the edge across
#   samples so this remains a visible geometric rule.

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


def _turn_delta(edge: str) -> tuple[int, int]:
    return {
        "left": (1, 0),
        "right": (-1, 0),
        "top": (0, 1),
        "bottom": (0, -1),
    }[edge]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint periodic rays that make one right-angle turn at a companion dot.

    Input structure:
    - The background is the majority color, with sparse colored dots elsewhere.
    - One grid edge is the unique launcher: it contains more non-background dots
      than any other edge, and it determines whether rays start along rows or
      along columns.
    - Every active shooting line has exactly one repeated marker color appearing
      twice; one marker sits on the shooting edge and the second sits one period
      inward, so their separation gives the beat.
    - A third dot on the same beat is a companion: it is farther along the same
      row/column and marks the elbow where the ray turns into the orthogonal axis.

    Rule:
    - Detect the unique shooting edge.  Left/right edges launch along rows;
      top/bottom edges launch along columns.
    - For each active line, infer the step from the two marker dots and look for
      a different-colored dot on the same modular beat.
    - Without a companion, keep the original straight-ray behavior: fill the
      entire row/column on that beat with the marker color.
    - With a companion, repaint beat positions from the shooting edge to the
      companion using the companion color, then continue from that corner along
      the orthogonal axis using the same step.  The turn direction is fixed by
      the edge: left->down, right->up, top->right, bottom->left.

    Key insight:
    - The two repeated markers still define a 1-D periodic ray, but an aligned
      companion no longer acts as a stop-only wall.  It becomes the visible elbow
      of an L-shape, and the same period is replayed on the orthogonal axis from
      that corner.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    H, W = g.shape

    edge = _shooting_edge(g, bg)
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

        repeating = {color: ps for color, ps in colors.items() if len(ps) >= 2}
        assert len(repeating) == 1, (
            f"Line {li} ({'row' if by_row else 'col'}) has repeating colors {dict(repeating)}; "
            "expected exactly one marker color"
        )
        marker, marker_positions = next(iter(repeating.items()))
        assert len(marker_positions) == 2, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
            f"has {len(marker_positions)} dots (expected 2): {marker_positions}"
        )
        a, b = sorted(marker_positions)
        step = b - a
        offset = a % step

        companion_color = None
        companion_pos = None
        for other_color, other_positions in colors.items():
            if other_color == marker:
                continue
            aligned = [p for p in other_positions if p % step == offset]
            if aligned:
                assert len(aligned) == 1, (
                    f"Line {li} ({'row' if by_row else 'col'}) has multiple aligned "
                    f"companion positions for color {other_color}: {aligned}"
                )
                assert companion_color is None, (
                    f"Line {li} ({'row' if by_row else 'col'}) has multiple companion colors"
                )
                companion_color = other_color
                companion_pos = aligned[0]

        if companion_color is None:
            paint = marker
            lo, hi = 0, line_len - 1
            for p in range(lo, hi + 1):
                if p % step != offset:
                    continue
                rc = cell(li, p)
                cur = int(out[rc])
                if cur == bg or cur == marker:
                    out[rc] = paint
            continue

        paint = companion_color
        lo, hi = sorted((0 if edge in ("left", "top") else line_len - 1, companion_pos))
        for p in range(lo, hi + 1):
            if p % step != offset:
                continue
            rc = cell(li, p)
            cur = int(out[rc])
            if cur in (bg, marker, companion_color):
                out[rc] = paint

        dr, dc = _turn_delta(edge)
        if by_row:
            r, c = li, companion_pos
        else:
            r, c = companion_pos, li

        while 0 <= r < H and 0 <= c < W:
            cur = int(out[r, c])
            if cur in (bg, marker, companion_color):
                out[r, c] = paint
            r += dr * step
            c += dc * step

    return out


_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_STEP_RANGE = (1, 5)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")
_MAX_RETRIES = 400


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


def _companion_beats(edge: str, line_len: int, step: int) -> list[int]:
    if edge in ("left", "top"):
        inner = step
        return list(range(inner + step, line_len - 1, step))
    inner = line_len - 1 - step
    return list(range(inner - step, 0, -step))


def _build_candidate(edge: str, line: int, H: int, W: int) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]], int] | None:
    by_row = edge in ("left", "right")
    line_len = W if by_row else H
    room = _orth_room(edge, line, H, W)
    max_step = min(_STEP_RANGE[1], room, (line_len - 2) // 2)
    if max_step < _STEP_RANGE[0]:
        return None

    steps = list(range(_STEP_RANGE[0], max_step + 1))
    random.shuffle(steps)
    for step in steps:
        companion_positions = _companion_beats(edge, line_len, step)
        if not companion_positions:
            continue
        random.shuffle(companion_positions)
        for companion_pos in companion_positions:
            marker_color = random.choice(_COLORS)
            other_colors = [color for color in _COLORS if color != marker_color]
            companion_color = random.choice(other_colors)

            input_cells: list[tuple[int, int, int]]
            output_cells: list[tuple[int, int, int]] = []

            if edge == "left":
                input_cells = [
                    (line, 0, marker_color),
                    (line, step, marker_color),
                    (line, companion_pos, companion_color),
                ]
                for c in range(0, companion_pos + 1, step):
                    output_cells.append((line, c, companion_color))
                for r in range(line, H, step):
                    output_cells.append((r, companion_pos, companion_color))
            elif edge == "right":
                input_cells = [
                    (line, W - 1, marker_color),
                    (line, W - 1 - step, marker_color),
                    (line, companion_pos, companion_color),
                ]
                for c in range(W - 1, companion_pos - 1, -step):
                    output_cells.append((line, c, companion_color))
                for r in range(line, -1, -step):
                    output_cells.append((r, companion_pos, companion_color))
            elif edge == "top":
                input_cells = [
                    (0, line, marker_color),
                    (step, line, marker_color),
                    (companion_pos, line, companion_color),
                ]
                for r in range(0, companion_pos + 1, step):
                    output_cells.append((r, line, companion_color))
                for c in range(line, W, step):
                    output_cells.append((companion_pos, c, companion_color))
            else:
                input_cells = [
                    (H - 1, line, marker_color),
                    (H - 1 - step, line, marker_color),
                    (companion_pos, line, companion_color),
                ]
                for r in range(H - 1, companion_pos - 1, -step):
                    output_cells.append((r, line, companion_color))
                for c in range(line, -1, -step):
                    output_cells.append((companion_pos, c, companion_color))

            dedup = {(r, c): color for r, c, color in output_cells}
            output_cells = [(r, c, color) for (r, c), color in dedup.items()]
            return input_cells, output_cells, companion_pos
    return None


def _count_other_edge_dots(dots_rc: list[tuple[int, int]], edge: str,
                           H: int, W: int) -> dict[str, int]:
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


def _choose_dims_and_lines(edge: str) -> tuple[int, int, list[int]] | None:
    by_row = edge in ("left", "right")
    if by_row:
        H = random.randint(max(_ROWS_RANGE[0], 8), _ROWS_RANGE[1])
        W = random.randint(*_COLS_RANGE)
    else:
        H = random.randint(*_ROWS_RANGE)
        W = random.randint(max(_COLS_RANGE[0], 10), _COLS_RANGE[1])

    allowed = _allowed_lines(edge, H, W)
    max_active = min(9, (len(allowed) + 1) // 2)
    if max_active < 4:
        return None
    n_active = random.randint(4, max_active)
    picks = _spaced_indices(n_active, len(allowed))
    if not picks:
        return None
    active_lines = [allowed[i] for i in picks]
    return H, W, _line_order(edge, active_lines, H, W)


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    edge = random.choice(_EDGES)
    dims = _choose_dims_and_lines(edge)
    if dims is None:
        return None
    H, W, active_lines = dims

    inp = np.zeros((H, W), dtype=int)
    expected = np.zeros((H, W), dtype=int)
    occupied_input: set[tuple[int, int]] = set()
    occupied_output: set[tuple[int, int]] = set()
    turned_axes: set[int] = set()

    for line in active_lines:
        placed = False
        for _ in range(120):
            candidate = _build_candidate(edge, line, H, W)
            if candidate is None:
                break
            input_cells, output_cells, turn_axis = candidate
            if turn_axis in turned_axes:
                continue
            if any((r, c) in occupied_input or (r, c) in occupied_output for r, c, _ in input_cells):
                continue
            if any((r, c) in occupied_output or (r, c) in occupied_input for r, c, _ in output_cells):
                continue

            for r, c, color in input_cells:
                inp[r, c] = color
                occupied_input.add((r, c))
            for r, c, color in output_cells:
                expected[r, c] = color
                occupied_output.add((r, c))
            turned_axes.add(turn_axis)
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
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            inp, _ = result
            return inp
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
