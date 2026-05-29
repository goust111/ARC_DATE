# Variant of 16de56c4
# Dimension: D12 — 多步处理 (multi_step)
# Change: 单步处理——所有活跃行独立并行计算，无依赖关系... → 链式反应

from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D12 variant
# a) Concrete behavior:
# - The original task had one global shooting edge and every active line could be
#   solved in isolation. The variant keeps the same "two same-color markers define
#   a periodic ray" language, but changes the fill into a directed process.
# - A ray now starts at a specific source dot, passes through the second same-color
#   marker on that line, and keeps stepping forward with the same gap.
# - When that directed ray hits an aligned companion, the segment is recolored by
#   the companion color exactly as before, but the struck companion also becomes a
#   new source. The new source looks for a same-color marker pair on the
#   perpendicular line through that hit cell and emits its own ray. This creates a
#   row -> column -> row or column -> row -> column cascade.
# - Small trace:
#   original line: A at edge, A one step inside, B further on-beat -> fill that
#   row segment with B, stop there.
#   variant: the same row still fills with B, but the struck B is also one end of
#   a vertical B pair, so a new vertical ray starts from that cell, may hit C, and
#   then C starts a final horizontal/vertical ray.
#
# b) Complexity calibration:
# - The generator always includes one real relay chain of length 3 rays
#   (initial source + two activations). This is enough to make D12 visible without
#   turning the puzzle into a dense maze.
# - Other initial edge lines still follow the old marker-only / companion /
#   bystander templates so the task still reads as a variant of 16de56c4.
#
# c) Hidden assumptions in the original solver that break here:
# - "Process each non-empty shooting line independently" no longer holds because
#   some non-edge lines are dormant until a companion is hit.
# - "Every non-empty shooting line has a repeating color" no longer holds because
#   a perpendicular relay marker can create rows/cols containing only one visible
#   dot before activation.
# - "Fill span is min/max of markers and companion" only worked because the source
#   was fixed at an edge. Relay rays need a directed source->second-marker->forward
#   interpretation, otherwise a perpendicular relay would incorrectly paint
#   backward.
# - "Parallel order is irrelevant" no longer holds conceptually; the solver must
#   maintain an activation queue. The generator avoids overlapping chains so the
#   final color does not depend on arbitrary queue tie-breaks.
#
# d) Placement legality:
# - Relay chains use fresh rows and columns: row r1 hits column c1, which hits row
#   r2. That keeps each line's sparse dots unambiguous.
# - Simple edge lines are forbidden from placing dots on the relay column/row that
#   would accidentally create fake companions for the chain.
# - All non-source dots are kept off the non-shooting boundaries, so the shooting
#   edge remains uniquely identifiable from boundary counts.
#
# e) Visual observability:
# - Every sample contains one non-edge relay line that appears only after another
#   line is completed, so the change is visible in the output as a staircase-like
#   cascade rather than a set of unrelated parallel fills.
#
# f) Visual cleanliness:
# - Only consecutive rays share a cell, and that shared cell is exactly the
#   companion/source handoff. No other overlaps are allowed.
#
# g) Color inference:
# - The companion color is structurally marked as "the next source color" because
#   the same color immediately forms a 2-dot marker pair on the perpendicular line
#   through the hit cell. No color-ID ordering is used anywhere.


# --- Configuration ---
_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 7)
_STEP_RANGE = (1, 4)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_P_MARKER_ONLY = 0.35
_P_COMPANION = 0.35

_MAX_RETRIES = 400


def _cell(orientation: str, line: int, pos: int) -> tuple[int, int]:
    return (line, pos) if orientation == "row" else (pos, line)


def _spaced_indices(n: int, total: int) -> list[int]:
    """Choose n indices from [0, total) with at least one empty slot between them."""
    available = total - (n - 1)
    if available < n:
        return []
    picks = sorted(random.sample(range(available), n))
    return [picks[i] + i for i in range(n)]


def _line_colors(g: np.ndarray, orientation: str, line: int, bg: int) -> dict[int, list[int]]:
    line_len = g.shape[1] if orientation == "row" else g.shape[0]
    colors: dict[int, list[int]] = defaultdict(list)
    for pos in range(line_len):
        v = int(g[_cell(orientation, line, pos)])
        if v != bg:
            colors[v].append(pos)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill periodic rays from one edge, then relay them through struck companions.

    Input structure:
    - bg = most common color. Sparse dots still identify a unique shooting edge.
    - Edge-sourced lines keep the old pattern: one boundary dot plus one same-color
      interior marker define the step on a row or column.
    - Some aligned companions also serve as relay sources: the struck companion is
      one endpoint of a same-color 2-dot marker pair on the perpendicular line.
    - Off-beat bystanders may appear, but they never define a relay.

    Rule:
    - Find the unique shooting edge; left/right start horizontal rays, top/bottom
      start vertical rays.
    - A ray is directed: source dot -> second same-color marker -> continue with
      the same step in that direction.
    - The first different-colored dot encountered on-beat becomes the paint color
      and stops that ray at the hit cell.
    - When such a hit cell is also the endpoint of a same-color pair on the
      perpendicular line, enqueue a new ray from that cell and continue the chain.
      Without a companion, the ray simply fills to the boundary.

    Key insight: the old "period from two markers" cue is still local, but the
    affected companion is now also a stateful relay node; solving requires a BFS
    over activated sources rather than independent per-line filling.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    H, W = g.shape

    edge_counts = {
        "top": sum(1 for c in range(W) if int(g[0, c]) != bg),
        "bottom": sum(1 for c in range(W) if int(g[H - 1, c]) != bg),
        "left": sum(1 for r in range(H) if int(g[r, 0]) != bg),
        "right": sum(1 for r in range(H) if int(g[r, W - 1]) != bg),
    }
    max_count = max(edge_counts.values())
    assert max_count > 0, "No non-bg dots on any edge"
    winners = [edge for edge, count in edge_counts.items() if count == max_count]
    assert len(winners) == 1, f"Ambiguous shooting edge: {winners} (counts={edge_counts})"
    edge = winners[0]

    initial_orientation = "row" if edge in ("left", "right") else "col"
    edge_pos = 0 if edge in ("left", "top") else (W - 1 if initial_orientation == "row" else H - 1)

    pair_maps: dict[str, dict[tuple[int, int, int], int]] = {"row": {}, "col": {}}
    for r in range(H):
        for color, positions in _line_colors(g, "row", r, bg).items():
            if len(positions) == 2:
                a, b = sorted(positions)
                pair_maps["row"][(r, a, int(color))] = b
                pair_maps["row"][(r, b, int(color))] = a
    for c in range(W):
        for color, positions in _line_colors(g, "col", c, bg).items():
            if len(positions) == 2:
                a, b = sorted(positions)
                pair_maps["col"][(c, a, int(color))] = b
                pair_maps["col"][(c, b, int(color))] = a

    queue: deque[tuple[str, int, int, int]] = deque()
    if initial_orientation == "row":
        for r in range(H):
            for color, positions in _line_colors(g, "row", r, bg).items():
                if len(positions) == 2 and edge_pos in positions:
                    queue.append(("row", r, edge_pos, int(color)))
    else:
        for c in range(W):
            for color, positions in _line_colors(g, "col", c, bg).items():
                if len(positions) == 2 and edge_pos in positions:
                    queue.append(("col", c, edge_pos, int(color)))
    assert queue, "No edge-sourced marker pairs found"

    seen_sources: set[tuple[str, int, int, int]] = set()

    while queue:
        orientation, line, source_pos, marker = queue.popleft()
        state = (orientation, line, source_pos, marker)
        if state in seen_sources:
            continue
        seen_sources.add(state)

        second_pos = pair_maps[orientation].get((line, source_pos, marker))
        if second_pos is None:
            continue

        step = abs(int(second_pos) - int(source_pos))
        if step == 0:
            continue
        direction = 1 if second_pos > source_pos else -1
        line_len = W if orientation == "row" else H

        companion_pos = None
        companion_color = None
        pos = source_pos + direction * step
        while 0 <= pos < line_len:
            r, c = _cell(orientation, line, pos)
            cur = int(g[r, c])
            if cur != bg and cur != marker:
                companion_pos = int(pos)
                companion_color = int(cur)
                break
            pos += direction * step

        paint = marker if companion_color is None else companion_color
        pos = source_pos
        while 0 <= pos < line_len:
            if companion_pos is not None and direction * (pos - companion_pos) > 0:
                break
            r, c = _cell(orientation, line, pos)
            cur = int(out[r, c])
            if cur == bg or cur == marker:
                out[r, c] = int(paint)
            pos += direction * step

        if companion_pos is None:
            continue

        hit_r, hit_c = _cell(orientation, line, companion_pos)
        next_orientation = "col" if orientation == "row" else "row"
        if next_orientation == "row":
            if (hit_r, hit_c, paint) in pair_maps["row"]:
                queue.append(("row", hit_r, hit_c, int(paint)))
        else:
            if (hit_c, hit_r, paint) in pair_maps["col"]:
                queue.append(("col", hit_c, hit_r, int(paint)))

    return out


def _place_dot(inp: np.ndarray, r: int, c: int, color: int) -> None:
    cur = int(inp[r, c])
    if cur not in (0, color):
        raise ValueError(f"Cell {(r, c)} already holds {cur}, cannot place {color}")
    inp[r, c] = int(color)


def _build_simple_initial_line(
    inp: np.ndarray,
    edge: str,
    line: int,
    palette: list[int],
    reserved_positions: set[int],
) -> None:
    orientation = "row" if edge in ("left", "right") else "col"
    line_len = inp.shape[1] if orientation == "row" else inp.shape[0]
    source_pos = 0 if edge in ("left", "top") else line_len - 1
    direction = 1 if edge in ("left", "top") else -1

    max_step = min(_STEP_RANGE[1], line_len - 3)
    if max_step < 1:
        raise ValueError("Line too short for a simple initial ray")
    step = random.randint(1, max_step)
    second_pos = source_pos + direction * step
    if not (0 < second_pos < line_len - 1):
        raise ValueError("Second marker would land on a forbidden edge")

    marker = random.choice(palette)
    sr, sc = _cell(orientation, line, source_pos)
    rr, rc = _cell(orientation, line, second_pos)
    _place_dot(inp, sr, sc, marker)
    _place_dot(inp, rr, rc, marker)

    roll = random.random()
    if roll < _P_MARKER_ONLY:
        return

    other_colors = [color for color in palette if color != marker]
    if not other_colors:
        return

    if roll < _P_MARKER_ONLY + _P_COMPANION:
        beats = []
        pos = source_pos + 2 * direction * step
        while 0 < pos < line_len - 1:
            if pos not in reserved_positions:
                beats.append(pos)
            pos += direction * step
        if beats:
            pos = random.choice(beats)
            r, c = _cell(orientation, line, pos)
            _place_dot(inp, r, c, random.choice(other_colors))
        return

    offset = source_pos % step
    candidates = [
        pos
        for pos in range(1, line_len - 1)
        if pos not in reserved_positions and pos not in (source_pos, second_pos) and pos % step != offset
    ]
    if candidates:
        pos = random.choice(candidates)
        r, c = _cell(orientation, line, pos)
        _place_dot(inp, r, c, random.choice(other_colors))


def _build_row_chain(
    inp: np.ndarray,
    edge: str,
    active_rows: list[int],
    palette: list[int],
) -> tuple[int, list[int]]:
    H, W = inp.shape
    source_col = 0 if edge == "left" else W - 1
    direction = 1 if edge == "left" else -1

    chain_rows = list(active_rows)
    r1 = random.choice(active_rows)
    free_rows = [r for r in range(1, H - 1) if r not in active_rows]
    if len(free_rows) < 2:
        raise ValueError("Not enough spare rows for the relay chain")

    colors = random.sample(palette, 3)
    color_a, color_b, color_c = map(int, colors)

    for _ in range(200):
        step1 = random.randint(1, min(_STEP_RANGE[1], W - 4))
        second1 = source_col + direction * step1
        if not (0 < second1 < W - 1):
            continue

        chain_cols = []
        pos = source_col + 2 * direction * step1
        while 1 <= pos <= W - 2:
            if 0 < pos + direction < W - 1:
                chain_cols.append(pos)
            pos += direction * step1
        if not chain_cols:
            continue
        c1 = random.choice(chain_cols)

        choices = []
        for vdir in (-1, 1):
            for step2 in range(1, min(_STEP_RANGE[1], H - 3) + 1):
                r_mid = r1 + vdir * step2
                if not (1 <= r_mid <= H - 2) or r_mid in chain_rows:
                    continue
                pos2 = r1 + 2 * vdir * step2
                while 1 <= pos2 <= H - 2:
                    r2 = pos2
                    if r2 not in chain_rows and r2 != r_mid:
                        choices.append((step2, r_mid, r2))
                    pos2 += vdir * step2
        if not choices:
            continue
        step2, r_mid, r2 = random.choice(choices)

        step3_candidates = []
        for step3 in range(1, min(_STEP_RANGE[1], W - 2) + 1):
            second3 = c1 + direction * step3
            if 0 < second3 < W - 1:
                step3_candidates.append((step3, second3))
        if not step3_candidates:
            continue
        _step3, second3 = random.choice(step3_candidates)

        _place_dot(inp, r1, source_col, color_a)
        _place_dot(inp, r1, second1, color_a)
        _place_dot(inp, r1, c1, color_b)

        _place_dot(inp, r_mid, c1, color_b)
        _place_dot(inp, r2, c1, color_c)
        _place_dot(inp, r2, second3, color_c)

        return c1, [r1, r_mid, r2]

    raise ValueError("Failed to build a row-based relay chain")


def _build_col_chain(
    inp: np.ndarray,
    edge: str,
    active_cols: list[int],
    palette: list[int],
) -> tuple[int, list[int]]:
    H, W = inp.shape
    source_row = 0 if edge == "top" else H - 1
    direction = 1 if edge == "top" else -1

    chain_cols = list(active_cols)
    c1 = random.choice(active_cols)
    free_cols = [c for c in range(1, W - 1) if c not in active_cols]
    if len(free_cols) < 2:
        raise ValueError("Not enough spare cols for the relay chain")

    colors = random.sample(palette, 3)
    color_a, color_b, color_c = map(int, colors)

    for _ in range(200):
        step1 = random.randint(1, min(_STEP_RANGE[1], H - 4))
        second1 = source_row + direction * step1
        if not (0 < second1 < H - 1):
            continue

        chain_rows = []
        pos = source_row + 2 * direction * step1
        while 1 <= pos <= H - 2:
            if 0 < pos + direction < H - 1:
                chain_rows.append(pos)
            pos += direction * step1
        if not chain_rows:
            continue
        r1 = random.choice(chain_rows)

        choices = []
        for hdir in (-1, 1):
            for step2 in range(1, min(_STEP_RANGE[1], W - 3) + 1):
                c_mid = c1 + hdir * step2
                if not (1 <= c_mid <= W - 2) or c_mid in chain_cols:
                    continue
                pos2 = c1 + 2 * hdir * step2
                while 1 <= pos2 <= W - 2:
                    c2 = pos2
                    if c2 not in chain_cols and c2 != c_mid:
                        choices.append((step2, c_mid, c2))
                    pos2 += hdir * step2
        if not choices:
            continue
        step2, c_mid, c2 = random.choice(choices)

        step3_candidates = []
        for step3 in range(1, min(_STEP_RANGE[1], H - 2) + 1):
            second3 = r1 + direction * step3
            if 0 < second3 < H - 1:
                step3_candidates.append((step3, second3))
        if not step3_candidates:
            continue
        _step3, second3 = random.choice(step3_candidates)

        _place_dot(inp, source_row, c1, color_a)
        _place_dot(inp, second1, c1, color_a)
        _place_dot(inp, r1, c1, color_b)

        _place_dot(inp, r1, c_mid, color_b)
        _place_dot(inp, r1, c2, color_c)
        _place_dot(inp, second3, c2, color_c)

        return r1, [c1, c_mid, c2]

    raise ValueError("Failed to build a column-based relay chain")


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(_EDGES)
    by_row = edge in ("left", "right")
    palette = random.sample(_COLORS, random.randint(4, min(7, len(_COLORS))))

    if by_row:
        H = random.randint(10, _ROWS_RANGE[1])
        W = random.randint(12, _COLS_RANGE[1])
        max_initial = min(_ACTIVE_LINES_RANGE[1], (H - 1) // 2)
        if max_initial < _ACTIVE_LINES_RANGE[0]:
            return None
        n_initial = random.randint(_ACTIVE_LINES_RANGE[0], max_initial)
        active_rows0 = _spaced_indices(n_initial, H - 2)
        if not active_rows0:
            return None
        active_rows = [r + 1 for r in active_rows0]

        inp = np.zeros((H, W), dtype=int)
        try:
            relay_col, _chain_rows = _build_row_chain(inp, edge, active_rows, palette)
            for row in active_rows:
                if np.any(inp[row] != 0):
                    continue
                _build_simple_initial_line(inp, edge, row, palette, {relay_col})
        except ValueError:
            return None
    else:
        H = random.randint(9, _ROWS_RANGE[1])
        W = random.randint(12, _COLS_RANGE[1])
        max_initial = min(_ACTIVE_LINES_RANGE[1], (W - 1) // 2)
        if max_initial < _ACTIVE_LINES_RANGE[0]:
            return None
        n_initial = random.randint(_ACTIVE_LINES_RANGE[0], max_initial)
        active_cols0 = _spaced_indices(n_initial, W - 2)
        if not active_cols0:
            return None
        active_cols = [c + 1 for c in active_cols0]

        inp = np.zeros((H, W), dtype=int)
        try:
            relay_row, _chain_cols = _build_col_chain(inp, edge, active_cols, palette)
            for col in active_cols:
                if np.any(inp[:, col] != 0):
                    continue
                _build_simple_initial_line(inp, edge, col, palette, {relay_row})
        except ValueError:
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
