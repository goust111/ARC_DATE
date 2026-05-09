# Combo variant of 16de56c4 (index 8)
# combo_spec: D5/0+D12/0
# Name: 碰撞停止+链式反应

from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D5/0 + D12/0 combo
# -------------------------------------------
# a) Concrete behavior and trace
#    The original task treated an aligned extra dot as a span endpoint and color
#    donor. This combo changes both pieces at once:
#    - D5: a ray travels physically away from its source edge and stops at the
#      first non-bg pixel ahead of it, even if that obstacle is off the periodic beat.
#    - D12: if the struck obstacle is also one endpoint of a same-color marker pair
#      on the perpendicular line, that obstacle becomes a new source and launches
#      a new periodic ray in its own color.
#    Small row-based trace:
#      input row r1: A . A . . B . . .
#      step(A)=2, B is off-beat at col 5 -> row ray paints cols 0,2,4 in A and
#      stops because col 5 is the first physical obstacle.
#      If column 5 also contains B again at row r_mid, then B is not just a wall:
#      it becomes a vertical source, so a new B-ray starts from (r1,5), passes
#      through (r_mid,5), and continues until its own first collision.
#    The crucial combo-specific difference is that relay activation no longer
#    requires the hit cell to lie on the old beat. An off-beat collision can now
#    both stop one ray and seed the next one.
#
# b) Complexity calibration
#    The generator keeps the object language sparse and repeated:
#    - exactly one guaranteed relay chain of three rays,
#    - exactly one passive aligned blocker line,
#    - exactly one passive off-beat blocker line,
#    - exactly one clear line with no blocker.
#    This is enough to make both D5 and D12 visible in every sample without
#    creating several competing cascades that would make the puzzle noisy.
#
# c) Original assumptions that break under the combo
#    - Independent per-line solving is invalid: a hit may activate a dormant
#      perpendicular line, so we need a BFS queue over sources.
#    - "Only aligned companions matter" is invalid: an off-beat obstacle can stop
#      a ray and can still be the relay handoff.
#    - "The hit recolors the whole segment" is invalid: under collision-stop the
#      current ray keeps its own marker color; the struck obstacle only seeds the
#      next ray.
#    - "Every non-empty line has a repeating color" is invalid: relay-support rows
#      or columns may contain only one visible dot until another ray reaches them.
#
# d) Placement legality guaranteed by the generator
#    - The initial edge lines are chosen on interior rows/cols so the shooting edge
#      stays uniquely identifiable.
#    - The relay column/row is reserved from all unrelated dots, preventing
#      accidental early collisions on the chain path.
#    - Colors are globally unique across unrelated structures; only intended
#      two-dot marker pairs reuse a color. This prevents accidental perpendicular
#      source pairs.
#    - The only overlaps are deliberate handoff cells where one ray's blocker is
#      simultaneously the next ray's source.
#
# e) Visual observability
#    Each sample always shows all four regimes side by side:
#    - a line that fills to the boundary,
#    - a line that stops at an aligned passive blocker,
#    - a line that stops at an off-beat passive blocker,
#    - a line that stops at an off-beat blocker and then triggers a perpendicular
#      relay chain.
#    That makes the fused rule inferable from a few examples without hidden state.
#
# f) Visual cleanliness
#    The cascade is a single staircase-like chain. No two unrelated rays share a
#    corridor, and no cell is overwritten by several different colors in the final
#    output. The only shared cells are the relay handoffs themselves.
#
# g) New solver assumptions checked against the generator
#    - A source is always one endpoint of an exact two-dot same-color pair on its
#      current row/column; generator enforces that for the edge ray and every relay.
#    - The current ray's direction is determined locally by source -> second marker;
#      generator builds all relay pairs with a consistent forward continuation.
#    - A blocker may or may not activate a new source; generator guarantees at
#      least one blocker does and at least two blockers do not.


_ROWS_RANGE = (9, 12)
_COLS_RANGE = (11, 21)
_STEP_RANGE = (1, 4)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")
_MAX_RETRIES = 500


def _cell(orientation: str, line: int, pos: int) -> tuple[int, int]:
    return (line, pos) if orientation == "row" else (pos, line)


def _forward_positions(start: int, direction: int, line_len: int) -> list[int]:
    out: list[int] = []
    pos = start + direction
    while 0 <= pos < line_len:
        out.append(pos)
        pos += direction
    return out


def _spaced_indices(n: int, total: int) -> list[int]:
    available = total - (n - 1)
    if available < n:
        return []
    picks = sorted(random.sample(range(available), n))
    return [picks[i] + i for i in range(n)]


def _line_colors(g: np.ndarray, orientation: str, line: int, bg: int) -> dict[int, list[int]]:
    line_len = g.shape[1] if orientation == "row" else g.shape[0]
    colors: dict[int, list[int]] = defaultdict(list)
    for pos in range(line_len):
        value = int(g[_cell(orientation, line, pos)])
        if value != bg:
            colors[value].append(pos)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch periodic rays from one edge, stop them on collision, and relay from hit nodes.

    Input structure:
    - The background is the most common color and all non-bg content is sparse.
    - One boundary has strictly more non-bg dots than the other three; that unique
      edge provides the initial sources (left/right -> rows, top/bottom -> columns).
    - Every real source is encoded by exactly two same-color dots on one line:
      the source cell and a second marker whose gap defines the periodic step.
    - Some obstacle dots also belong to a same-color pair on the perpendicular line;
      those are dormant relay sources until a ray physically collides with them.

    Rule:
    - Find the unique shooting edge and enqueue every two-dot marker pair that starts there.
    - For a queued source, use the source-to-second-marker direction and gap as the ray's beat.
    - Scan forward cell by cell; the first non-background pixel ahead is a hard stop,
      even if it is off the beat. Paint only the source color on beat positions strictly
      before that obstacle. If there is no obstacle, paint to the boundary.
    - When the stopping obstacle is also one endpoint of a same-color pair on the
      perpendicular line, enqueue that obstacle as a new source and continue in BFS order.

    Key insight: the local "two equal markers define the step" cue still applies, but
    blockers now have two possible roles: passive walls or relay handoffs. The cascade
    is discovered by following collisions, not by solving each line independently.
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
    initial_edge_pos = 0 if edge in ("left", "top") else (W - 1 if initial_orientation == "row" else H - 1)

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
                if len(positions) == 2 and initial_edge_pos in positions:
                    queue.append(("row", r, initial_edge_pos, int(color)))
    else:
        for c in range(W):
            for color, positions in _line_colors(g, "col", c, bg).items():
                if len(positions) == 2 and initial_edge_pos in positions:
                    queue.append(("col", c, initial_edge_pos, int(color)))
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
        assert step > 0, f"Zero step at source {state}"
        direction = 1 if second_pos > source_pos else -1
        line_len = W if orientation == "row" else H

        hit_pos = None
        hit_color = None
        pos = second_pos + direction
        while 0 <= pos < line_len:
            r, c = _cell(orientation, line, pos)
            cur = int(g[r, c])
            if cur != bg:
                hit_pos = int(pos)
                hit_color = cur
                break
            pos += direction

        pos = source_pos
        while 0 <= pos < line_len:
            if hit_pos is not None and direction * (pos - hit_pos) >= 0:
                break
            r, c = _cell(orientation, line, pos)
            cur = int(out[r, c])
            if cur == bg or cur == marker:
                out[r, c] = int(marker)
            pos += direction * step

        if hit_pos is None or hit_color is None:
            continue

        hit_r, hit_c = _cell(orientation, line, hit_pos)
        next_orientation = "col" if orientation == "row" else "row"
        if next_orientation == "row":
            if (hit_r, hit_c, hit_color) in pair_maps["row"]:
                queue.append(("row", hit_r, hit_c, int(hit_color)))
        else:
            if (hit_c, hit_r, hit_color) in pair_maps["col"]:
                queue.append(("col", hit_c, hit_r, int(hit_color)))

    return out


def _place_dot(inp: np.ndarray, r: int, c: int, color: int) -> None:
    cur = int(inp[r, c])
    if cur not in (0, color):
        raise ValueError(f"Cell {(r, c)} already holds {cur}, cannot place {color}")
    inp[r, c] = int(color)


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


def _build_simple_initial_line(
    inp: np.ndarray,
    edge: str,
    line: int,
    marker_color: int,
    blocker_color: int | None,
    line_type: str,
    forbidden_positions: set[int],
) -> None:
    orientation = "row" if edge in ("left", "right") else "col"
    line_len = inp.shape[1] if orientation == "row" else inp.shape[0]
    source_pos = 0 if edge in ("left", "top") else line_len - 1
    direction = 1 if edge in ("left", "top") else -1

    for _ in range(200):
        step = random.randint(_STEP_RANGE[0], min(_STEP_RANGE[1], line_len - 3))
        second_pos = source_pos + direction * step
        if not (0 < second_pos < line_len - 1):
            continue
        if second_pos in forbidden_positions:
            continue

        future = [
            pos
            for pos in _forward_positions(second_pos, direction, line_len)
            if 0 < pos < line_len - 1 and pos not in forbidden_positions
        ]
        min_visible_fill = source_pos + 2 * direction * step
        future = [pos for pos in future if direction * (pos - min_visible_fill) > 0]
        beat = [pos for pos in future if (pos - source_pos) % step == 0]
        offbeat = [pos for pos in future if (pos - source_pos) % step != 0]

        if line_type == "clear":
            blocker_pos = None
        elif line_type == "aligned_blocker":
            if not beat or blocker_color is None:
                continue
            blocker_pos = random.choice(beat)
        elif line_type == "offbeat_blocker":
            if not offbeat or blocker_color is None:
                continue
            blocker_pos = random.choice(offbeat)
        else:
            raise ValueError(f"Unknown line_type: {line_type}")

        sr, sc = _cell(orientation, line, source_pos)
        mr, mc = _cell(orientation, line, second_pos)
        _place_dot(inp, sr, sc, marker_color)
        _place_dot(inp, mr, mc, marker_color)
        if blocker_pos is not None and blocker_color is not None:
            br, bc = _cell(orientation, line, blocker_pos)
            _place_dot(inp, br, bc, blocker_color)
        return

    raise ValueError(f"Failed to build simple line {line} ({line_type})")


def _build_row_chain(
    inp: np.ndarray,
    edge: str,
    active_rows: list[int],
    colors: list[int],
) -> tuple[int, list[tuple[int, int]]]:
    H, W = inp.shape
    source_col = 0 if edge == "left" else W - 1
    hdir = 1 if edge == "left" else -1
    color_a, color_b, color_c = map(int, colors)

    r1 = random.choice(active_rows)
    occupied_rows = set(active_rows)

    for _ in range(400):
        step1 = random.randint(1, min(_STEP_RANGE[1], W - 4))
        second1 = source_col + hdir * step1
        if not (0 < second1 < W - 1):
            continue

        relay_cols = [
            c1
            for c1 in _forward_positions(second1, hdir, W)
            if 0 < c1 < W - 1
            and (c1 - source_col) % step1 != 0
            and hdir * (c1 - (source_col + 2 * hdir * step1)) > 0
        ]
        random.shuffle(relay_cols)
        for c1 in relay_cols:
            vertical_choices: list[tuple[int, int, int, str]] = []
            for vdir in (-1, 1):
                for step2 in range(1, min(_STEP_RANGE[1], H - 3) + 1):
                    r_mid = r1 + vdir * step2
                    if not (1 <= r_mid <= H - 2):
                        continue
                    if r_mid in occupied_rows:
                        continue
                    for mode in ("onbeat", "offbeat"):
                        hits = [
                            r2
                            for r2 in _forward_positions(r_mid, vdir, H)
                            if 1 <= r2 <= H - 2
                            and r2 not in occupied_rows
                            and r2 != r_mid
                            and vdir * (r2 - (r1 + 2 * vdir * step2)) > 0
                            and (((r2 - r1) % step2 == 0) if mode == "onbeat" else ((r2 - r1) % step2 != 0))
                        ]
                        if hits:
                            vertical_choices.append((r_mid, random.choice(hits), step2, mode))
            if not vertical_choices:
                continue

            random.shuffle(vertical_choices)
            for r_mid, r2, _step2, _mode2 in vertical_choices:
                step3_candidates = [
                    step3
                    for step3 in range(1, min(_STEP_RANGE[1], W - 2) + 1)
                    if 0 < c1 + hdir * step3 < W - 1 and 0 <= c1 + 2 * hdir * step3 < W
                ]
                if not step3_candidates:
                    continue
                step3 = random.choice(step3_candidates)
                c2 = c1 + hdir * step3

                _place_dot(inp, r1, source_col, color_a)
                _place_dot(inp, r1, second1, color_a)
                _place_dot(inp, r1, c1, color_b)
                _place_dot(inp, r_mid, c1, color_b)
                _place_dot(inp, r2, c1, color_c)
                _place_dot(inp, r2, c2, color_c)

                return c1, [(r1, c1), (r_mid, c1), (r2, c1)]

    raise ValueError("Failed to build row-based relay chain")


def _build_col_chain(
    inp: np.ndarray,
    edge: str,
    active_cols: list[int],
    colors: list[int],
) -> tuple[int, list[tuple[int, int]]]:
    H, W = inp.shape
    source_row = 0 if edge == "top" else H - 1
    vdir = 1 if edge == "top" else -1
    color_a, color_b, color_c = map(int, colors)

    c1 = random.choice(active_cols)
    occupied_cols = set(active_cols)

    for _ in range(400):
        step1 = random.randint(1, min(_STEP_RANGE[1], H - 4))
        second1 = source_row + vdir * step1
        if not (0 < second1 < H - 1):
            continue

        relay_rows = [
            r1
            for r1 in _forward_positions(second1, vdir, H)
            if 0 < r1 < H - 1
            and (r1 - source_row) % step1 != 0
            and vdir * (r1 - (source_row + 2 * vdir * step1)) > 0
        ]
        random.shuffle(relay_rows)
        for r1 in relay_rows:
            horizontal_choices: list[tuple[int, int, int, str]] = []
            for hdir in (-1, 1):
                for step2 in range(1, min(_STEP_RANGE[1], W - 3) + 1):
                    c_mid = c1 + hdir * step2
                    if not (1 <= c_mid <= W - 2):
                        continue
                    if c_mid in occupied_cols:
                        continue
                    for mode in ("onbeat", "offbeat"):
                        hits = [
                            c2
                            for c2 in _forward_positions(c_mid, hdir, W)
                            if 1 <= c2 <= W - 2
                            and c2 not in occupied_cols
                            and c2 != c_mid
                            and hdir * (c2 - (c1 + 2 * hdir * step2)) > 0
                            and (((c2 - c1) % step2 == 0) if mode == "onbeat" else ((c2 - c1) % step2 != 0))
                        ]
                        if hits:
                            horizontal_choices.append((c_mid, random.choice(hits), step2, mode))
            if not horizontal_choices:
                continue

            random.shuffle(horizontal_choices)
            for c_mid, c2, _step2, _mode2 in horizontal_choices:
                step3_candidates = [
                    step3
                    for step3 in range(1, min(_STEP_RANGE[1], H - 2) + 1)
                    if 0 < r1 + vdir * step3 < H - 1 and 0 <= r1 + 2 * vdir * step3 < H
                ]
                if not step3_candidates:
                    continue
                step3 = random.choice(step3_candidates)
                r2 = r1 + vdir * step3

                _place_dot(inp, source_row, c1, color_a)
                _place_dot(inp, second1, c1, color_a)
                _place_dot(inp, r1, c1, color_b)
                _place_dot(inp, r1, c_mid, color_b)
                _place_dot(inp, r1, c2, color_c)
                _place_dot(inp, r2, c2, color_c)

                return r1, [(r1, c1), (r1, c_mid), (r1, c2)]

    raise ValueError("Failed to build col-based relay chain")


def _gen_attempt() -> np.ndarray | None:
    edge = random.choice(_EDGES)
    by_row = edge in ("left", "right")

    H = random.randint(*_ROWS_RANGE)
    W = random.randint(*_COLS_RANGE)

    n_active = 4
    if by_row:
        interior_rows = H - 2
        base = _spaced_indices(n_active, interior_rows)
        if not base:
            return None
        active_rows = [r + 1 for r in base]
    else:
        interior_cols = W - 2
        base = _spaced_indices(n_active, interior_cols)
        if not base:
            return None
        active_cols = [c + 1 for c in base]

    colors = random.sample(_COLORS, 9)
    chain_colors = colors[:3]
    simple_colors = colors[3:]

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []

    try:
        if by_row:
            reserved_col, _chain = _build_row_chain(inp, edge, active_rows, chain_colors)
            other_rows = [r for r in active_rows if not np.any(inp[r] != 0)]
            line_plan = [
                ("clear", simple_colors[0], None),
                ("aligned_blocker", simple_colors[1], simple_colors[2]),
                ("offbeat_blocker", simple_colors[3], simple_colors[4]),
            ]
            random.shuffle(line_plan)
            for row, (line_type, marker_color, blocker_color) in zip(other_rows, line_plan):
                _build_simple_initial_line(
                    inp,
                    edge,
                    row,
                    marker_color,
                    blocker_color,
                    line_type,
                    {reserved_col},
                )
        else:
            reserved_row, _chain = _build_col_chain(inp, edge, active_cols, chain_colors)
            other_cols = [c for c in active_cols if not np.any(inp[:, c] != 0)]
            line_plan = [
                ("clear", simple_colors[0], None),
                ("aligned_blocker", simple_colors[1], simple_colors[2]),
                ("offbeat_blocker", simple_colors[3], simple_colors[4]),
            ]
            random.shuffle(line_plan)
            for col, (line_type, marker_color, blocker_color) in zip(other_cols, line_plan):
                _build_simple_initial_line(
                    inp,
                    edge,
                    col,
                    marker_color,
                    blocker_color,
                    line_type,
                    {reserved_row},
                )
    except ValueError:
        return None

    for r in range(H):
        for c in range(W):
            if int(inp[r, c]) != 0:
                all_rc.append((r, c))

    shooting_edge_dots = len(active_rows) if by_row else len(active_cols)
    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(count >= shooting_edge_dots for count in other_counts.values()):
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
