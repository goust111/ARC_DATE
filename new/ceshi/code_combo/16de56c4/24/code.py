# Combo variant of 16de56c4 (index 24)
# combo_spec: D2/0+D12/0
# Name: 每行独立方向+链式反应

from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np

from arc_utils import get_bg


# Design analysis
# a) Concrete behavior and trace:
#    - The fused task keeps one global starting axis at a time (rows or columns),
#      but removes any global side. Every source line chooses its own low/high
#      side, encoded by two off-beat guide dots that sit one cell away from that
#      source and one step apart.
#    - Initial sources are implicit boundary cells. Example on a row of length 12:
#      left-sourced step-3 line: guides at 1 and 4, beat cells at 0,3,6,9...
#      right-sourced step-3 line: guides at 7 and 10, beat cells at 11,8,5,2...
#      This is the D2 part: the same gap can generate different painted classes
#      depending on which side the guides hug.
#    - Chain reaction is added by letting an on-beat companion become a new source
#      on the perpendicular line through the hit cell. That new source is not
#      boundary-anchored; instead the hit cell itself is the source cell, and two
#      same-color guide dots are placed immediately on one chosen side of it.
#    - Small trace:
#      row source A from the left: A-guides at c=1,4, companion B at c=6.
#      Output paints row beats 0,3,6 with B.
#      The struck B at (r,6) is also a column source, say top-sourced: B-guides
#      at r+1 and r+4, companion C at r+6. That paints the column beats with C.
#      The struck C then starts a final row source whose side is chosen locally.
#    - Difference from the original task: there is no single shooting edge for all
#      lines, the repeated pair is off the painted progression, and some lines stay
#      dormant until a previous hit activates them.
# b) Complexity calibration:
#    - The generator always includes one real 3-ray cascade
#      (initial boundary source -> relay -> relay) so the multi-step behavior is
#      visible in every sample.
#    - Among the initial boundary-sourced lines, both opposite sides of the global
#      axis are forced to appear. This makes the D2 local-direction signal visible
#      instead of collapsing to one uniform side.
#    - Extra lines stay sparse and use only the original marker-only / companion /
#      bystander vocabulary, so the task still reads as a close relative of the
#      base puzzle instead of becoming a dense maze.
# c) Hidden assumptions from the original solver that break here:
#    - Counting boundary dots cannot recover the source side, because initial guide
#      dots are one cell off the boundary and relay sources are internal.
#    - A repeated color of size 2 is no longer enough to describe every source:
#      relay sources use one visible source cell plus two same-color guide dots.
#    - Lines are not independent. The solver needs a queue of activated sources.
#    - The modular class of the painted cells is not the modular class of the guide
#      dots. It is the class adjacent to them on the source side.
# d) Placement legality:
#    - Initial active lines live on one global axis only and are spaced apart.
#    - Relay lines use fresh rows/cols on the perpendicular axis, and the only
#      shared cells are the handoff cells where one companion becomes the next
#      source.
#    - Simple initial lines are forbidden from placing dots on the relay column or
#      relay row, so accidental earlier companions cannot corrupt the cascade.
# e) Visual observability:
#    - In each sample, one can see opposite-side initial rows/cols filling
#      different boundary-anchored beat classes, and at least one non-initial
#      perpendicular line appears only after a hit. That makes both ingredients
#      inferable from 2-3 examples.
# f) Visual cleanliness:
#    - The chain occupies a thin staircase: one row, one column, then one row
#      again (or the transposed version). Other lines avoid the relay corridor.
#      No two unrelated structures pile onto the same cells.
# g) Color inference:
#    - Colors are structural, not numeric. A guide color means "this source would
#      keep propagating with this period"; a companion color means "recolor this
#      segment and possibly spawn the next source". Relay colors are repeated on
#      their guide pair immediately next to the handoff cell, which gives a visible
#      cue for why that color becomes the next source.
# h) Variant metric used during validation:
#    - `relay_count`: number of activated non-boundary sources; every valid sample
#      must have at least 2 (the middle and final rays of the built chain).
#    - `initial_side_count`: number of distinct low/high sides among boundary
#      sources on the chosen axis; every valid sample must be 2.
# i) New solver assumptions audited against the generator:
#    - Boundary sources are recognized only from guide pairs at `[1, 1+step]` or
#      `[L-2-step, L-2]`. The generator emits only those for initial lines.
#    - Relay sources are recognized only when the hit cell itself has the relay
#      color and the same line also has exactly two adjacent-side guide dots at
#      `source+dir` and `source+dir*(1+step)`. The generator enforces that.
#    - The first on-beat different-color dot beyond the farther guide is the only
#      valid companion. Generator placement reserves the relay corridor so nothing
#      else can appear earlier on those beats.


_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 7)
_STEP_RANGE = (2, 5)
_COLORS = list(range(1, 10))

_P_MARKER_ONLY = 0.35
_P_COMPANION = 0.40

_MAX_RETRIES = 600


def _cell(orientation: str, line: int, pos: int) -> tuple[int, int]:
    return (line, pos) if orientation == "row" else (pos, line)


def _line_len(g: np.ndarray, orientation: str) -> int:
    return g.shape[1] if orientation == "row" else g.shape[0]


def _line_colors(g: np.ndarray, orientation: str, line: int, bg: int) -> dict[int, list[int]]:
    line_len = _line_len(g, orientation)
    colors: dict[int, list[int]] = defaultdict(list)
    for pos in range(line_len):
        v = int(g[_cell(orientation, line, pos)])
        if v != bg:
            colors[v].append(pos)
    return colors


def _boundary_source_from_line(
    g: np.ndarray, bg: int, orientation: str, line: int
) -> dict[str, int | str] | None:
    line_len = _line_len(g, orientation)
    colors = _line_colors(g, orientation, line, bg)
    if not colors:
        return None

    matches: list[dict[str, int | str]] = []
    for color, positions in colors.items():
        if len(positions) != 2:
            continue
        a, b = sorted(positions)
        step = b - a
        if step < _STEP_RANGE[0]:
            continue
        if a == 1 and b == 1 + step:
            matches.append(
                {
                    "orientation": orientation,
                    "line": line,
                    "source_pos": 0,
                    "marker": int(color),
                    "step": step,
                    "direction": 1,
                    "kind": "boundary",
                }
            )
        if b == line_len - 2 and a == line_len - 2 - step:
            matches.append(
                {
                    "orientation": orientation,
                    "line": line,
                    "source_pos": line_len - 1,
                    "marker": int(color),
                    "step": step,
                    "direction": -1,
                    "kind": "boundary",
                }
            )

    if not matches:
        return None
    assert len(matches) == 1, f"Line {line} ({orientation}) has ambiguous boundary guides: {matches}"
    return matches[0]


def _relay_source_from_hit(
    g: np.ndarray, bg: int, orientation: str, line: int, source_pos: int, marker: int
) -> dict[str, int | str] | None:
    r, c = _cell(orientation, line, source_pos)
    assert int(g[r, c]) == marker, (
        f"Relay source cell {(r, c)} on {orientation} {line} should hold color {marker}"
    )

    positions = sorted(_line_colors(g, orientation, line, bg).get(marker, []))
    if len(positions) != 3 or source_pos not in positions:
        return None

    for direction in (-1, 1):
        p1 = source_pos + direction
        if p1 not in positions:
            continue
        for p2 in positions:
            if p2 in (source_pos, p1):
                continue
            delta = direction * (p2 - p1)
            if delta < _STEP_RANGE[0]:
                continue
            if p2 != source_pos + direction * (delta + 1):
                continue
            return {
                "orientation": orientation,
                "line": line,
                "source_pos": source_pos,
                "marker": marker,
                "step": delta,
                "direction": direction,
                "kind": "relay",
            }
    return None


def _collect_boundary_sources(g: np.ndarray, bg: int, orientation: str) -> list[dict[str, int | str]]:
    num_lines = g.shape[0] if orientation == "row" else g.shape[1]
    sources: list[dict[str, int | str]] = []
    for line in range(num_lines):
        info = _boundary_source_from_line(g, bg, orientation, line)
        if info is not None:
            sources.append(info)
    return sources


def _paint_ray(
    g: np.ndarray,
    out: np.ndarray,
    bg: int,
    info: dict[str, int | str],
) -> tuple[int | None, int | None, int | None]:
    orientation = str(info["orientation"])
    line = int(info["line"])
    source_pos = int(info["source_pos"])
    marker = int(info["marker"])
    step = int(info["step"])
    direction = int(info["direction"])
    line_len = _line_len(g, orientation)
    far_guide = source_pos + direction * (step + 1)

    companion_pos = None
    companion_color = None
    pos = source_pos + direction * (2 * step)
    while 0 <= pos < line_len:
        r, c = _cell(orientation, line, pos)
        cur = int(g[r, c])
        if cur != bg and cur != marker:
            companion_pos = pos
            companion_color = cur
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
            out[r, c] = paint
        pos += direction * step

    if companion_pos is not None:
        assert direction * (companion_pos - far_guide) > 0, (
            f"Companion on {orientation} {line} appears before the far guide"
        )

    return companion_pos, companion_color, paint


def _simulate(
    grid: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    row_sources = _collect_boundary_sources(g, bg, "row")
    col_sources = _collect_boundary_sources(g, bg, "col")
    assert row_sources or col_sources, "No boundary-anchored guide lines found"

    if row_sources and col_sources:
        assert len(row_sources) != len(col_sources), (
            f"Ambiguous starting axis: rows={len(row_sources)}, cols={len(col_sources)}"
        )
    initial_sources = row_sources if len(row_sources) > len(col_sources) else col_sources
    initial_orientation = str(initial_sources[0]["orientation"])

    queue: deque[dict[str, int | str]] = deque(initial_sources)
    seen: set[tuple[str, int, int, int]] = set()
    processed: list[dict[str, int | str]] = []
    relay_count = 0

    while queue:
        info = queue.popleft()
        state = (
            str(info["orientation"]),
            int(info["line"]),
            int(info["source_pos"]),
            int(info["marker"]),
        )
        if state in seen:
            continue
        seen.add(state)
        processed.append(info)
        if str(info["kind"]) == "relay":
            relay_count += 1

        companion_pos, companion_color, paint = _paint_ray(g, out, bg, info)
        if companion_pos is None or companion_color is None:
            continue

        next_orientation = "col" if str(info["orientation"]) == "row" else "row"
        if next_orientation == "row":
            next_line = int(_cell(str(info["orientation"]), int(info["line"]), companion_pos)[0])
            next_source_pos = int(_cell(str(info["orientation"]), int(info["line"]), companion_pos)[1])
        else:
            next_source_pos = int(_cell(str(info["orientation"]), int(info["line"]), companion_pos)[0])
            next_line = int(_cell(str(info["orientation"]), int(info["line"]), companion_pos)[1])
        relay = _relay_source_from_hit(g, bg, next_orientation, next_line, next_source_pos, paint)
        if relay is not None:
            queue.append(relay)

    initial_sides = {int(info["direction"]) for info in initial_sources}
    meta = {
        "initial_orientation": initial_orientation,
        "initial_count": len(initial_sources),
        "initial_side_count": len(initial_sides),
        "relay_count": relay_count,
        "processed": processed,
    }
    return out, meta


def analyze_sample(grid: np.ndarray) -> dict[str, object]:
    _, meta = _simulate(grid)
    return meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill locally directed periodic rays, then relay them through struck companions.

    Input structure:
    - Background is the most common color and the non-background content is sparse.
    - One global starting axis is used at a time: either several rows or several
      columns contain boundary-sourced guide pairs, but each such line chooses its
      own side independently.
    - A boundary source is encoded by exactly two same-color guide dots placed one
      cell away from the chosen edge and one step apart.
    - Some on-beat companions also encode relay sources: the struck cell itself
      has that companion color, and on the perpendicular line there are two more
      same-color guide dots immediately on one chosen side of it.

    Rule:
    - First identify whether rows or columns contain more legal boundary-sourced
      guide pairs; that is the starting axis.
    - For any source, the guide-dot gap is the step. The painted progression is
      not on the guide dots themselves; it is the beat class adjacent to them on
      the source side.
    - March along that beat progression in the encoded direction. The first
      different-colored dot encountered beyond the farther guide is a companion:
      it recolors the whole traversed beat segment and stops that ray.
    - When a companion is hit, check the perpendicular line through that cell. If
      the hit cell plus two same-color guide dots form a legal relay source there,
      enqueue it and continue the cascade. Dots off the beat progression are
      bystanders and remain unchanged.

    Key insight:
    - Two distinct cues must be fused: guide-dot placement tells you the local
      direction of each ray, and a hit companion can only be understood by seeing
      it as the explicit source of a new perpendicular ray.
    """
    out, _meta = _simulate(grid)
    return out


def _spaced_indices(n: int, total: int) -> list[int]:
    available = total - (n - 1)
    if available < n:
        return []
    picks = sorted(random.sample(range(available), n))
    return [picks[i] + i for i in range(n)]


def _place_dot(inp: np.ndarray, r: int, c: int, color: int) -> None:
    cur = int(inp[r, c])
    if cur not in (0, color):
        raise ValueError(f"Cell {(r, c)} already holds {cur}, cannot place {color}")
    inp[r, c] = int(color)


def _pick_step(max_step: int) -> int:
    if max_step < _STEP_RANGE[0]:
        raise ValueError("No legal step available")
    return random.randint(_STEP_RANGE[0], max_step)


def _place_boundary_guides(
    inp: np.ndarray,
    orientation: str,
    line: int,
    side: str,
    color: int,
    step: int,
) -> tuple[int, int, int]:
    line_len = _line_len(inp, orientation)
    if side in ("left", "top"):
        source_pos = 0
        direction = 1
        p1 = 1
        p2 = 1 + step
    else:
        source_pos = line_len - 1
        direction = -1
        p2 = line_len - 2
        p1 = line_len - 2 - step

    r1, c1 = _cell(orientation, line, p1)
    r2, c2 = _cell(orientation, line, p2)
    _place_dot(inp, r1, c1, color)
    _place_dot(inp, r2, c2, color)
    return source_pos, direction, p2 if direction > 0 else p1


def _place_relay_guides(
    inp: np.ndarray,
    orientation: str,
    line: int,
    source_pos: int,
    side: str,
    color: int,
    step: int,
) -> int:
    direction = 1 if side in ("left", "top") else -1
    p1 = source_pos + direction
    p2 = source_pos + direction * (step + 1)
    r1, c1 = _cell(orientation, line, p1)
    r2, c2 = _cell(orientation, line, p2)
    _place_dot(inp, r1, c1, color)
    _place_dot(inp, r2, c2, color)
    return direction


def _build_simple_initial_line(
    inp: np.ndarray,
    orientation: str,
    line: int,
    side: str,
    palette: list[int],
    blocked_positions: set[int],
    require_companion: bool = False,
) -> str:
    line_len = _line_len(inp, orientation)
    marker = int(random.choice(palette))
    max_step = min(_STEP_RANGE[1], line_len - 3)

    candidates: list[int] = []
    for step in range(_STEP_RANGE[0], max_step + 1):
        if side in ("left", "top"):
            guide_positions = {1, 1 + step}
        else:
            guide_positions = {line_len - 2 - step, line_len - 2}
        if guide_positions.isdisjoint(blocked_positions):
            candidates.append(step)
    if not candidates:
        raise ValueError("No legal step that avoids the relay corridor")

    step = random.choice(candidates)
    source_pos, direction, far_guide = _place_boundary_guides(inp, orientation, line, side, marker, step)

    beats = [
        source_pos + direction * k * step
        for k in range(2, line_len)
        if 0 <= source_pos + direction * k * step < line_len
        and direction * (source_pos + direction * k * step - far_guide) > 0
        and source_pos + direction * k * step not in blocked_positions
    ]

    off_beat = [
        pos
        for pos in range(line_len)
        if pos not in blocked_positions
        and pos not in {1, 1 + step, line_len - 2, line_len - 2 - step}
        and (pos - source_pos) % step != 0
    ]

    if require_companion and not beats:
        raise ValueError("Requested companion but no legal beat exists")

    roll = random.random()
    if require_companion:
        roll = _P_MARKER_ONLY + 0.01

    other_colors = [c for c in palette if c != marker]
    if roll < _P_MARKER_ONLY or not other_colors:
        return "marker_only"
    if roll < _P_MARKER_ONLY + _P_COMPANION and beats:
        pos = random.choice(beats)
        r, c = _cell(orientation, line, pos)
        _place_dot(inp, r, c, int(random.choice(other_colors)))
        return "companion"
    if off_beat:
        pos = random.choice(off_beat)
        r, c = _cell(orientation, line, pos)
        _place_dot(inp, r, c, int(random.choice(other_colors)))
        return "bystander"
    return "marker_only"


def _build_row_chain(
    inp: np.ndarray,
    active_rows: list[int],
    palette: list[int],
) -> tuple[int, int]:
    H, W = inp.shape
    free_rows = [r for r in range(1, H - 1) if r not in active_rows]
    if len(free_rows) < 1:
        raise ValueError("Need a free row for the final relay")

    colors = random.sample(palette, 3)
    color_a, color_b, color_c = map(int, colors)
    r1 = random.choice(active_rows)

    for _ in range(500):
        side1 = random.choice(["left", "right"])
        dir1 = 1 if side1 == "left" else -1
        max_step1 = min(_STEP_RANGE[1], W - 3)
        step1 = _pick_step(max_step1)
        source1 = 0 if dir1 > 0 else W - 1
        far1 = source1 + dir1 * (step1 + 1)
        companion_cols = [
            source1 + dir1 * k * step1
            for k in range(2, W)
            if 1 <= source1 + dir1 * k * step1 <= W - 2
            and dir1 * (source1 + dir1 * k * step1 - far1) > 0
        ]
        if not companion_cols:
            continue
        c1 = random.choice(companion_cols)

        side2 = random.choice(["top", "bottom"])
        dir2 = 1 if side2 == "top" else -1
        max_step2 = min(_STEP_RANGE[1], H - 3)
        step2 = _pick_step(max_step2)
        guide1 = r1 + dir2
        guide2 = r1 + dir2 * (step2 + 1)
        final_rows = [
            r1 + dir2 * k * step2
            for k in range(2, H)
            if 1 <= r1 + dir2 * k * step2 <= H - 2
            and r1 + dir2 * k * step2 in free_rows
            and dir2 * (r1 + dir2 * k * step2 - guide2) > 0
        ]
        if not (
            1 <= guide1 <= H - 2
            and 1 <= guide2 <= H - 2
            and guide1 not in active_rows
            and guide2 not in active_rows
            and final_rows
        ):
            continue
        r2 = random.choice(final_rows)

        side3 = random.choice(["left", "right"])
        dir3 = 1 if side3 == "left" else -1
        max_step3 = min(_STEP_RANGE[1], W - 3)
        step3 = _pick_step(max_step3)
        guide3a = c1 + dir3
        guide3b = c1 + dir3 * (step3 + 1)
        if not (1 <= guide3a <= W - 2 and 1 <= guide3b <= W - 2):
            continue

        _place_boundary_guides(inp, "row", r1, side1, color_a, step1)
        _place_dot(inp, r1, c1, color_b)
        _place_relay_guides(inp, "col", c1, r1, side2, color_b, step2)
        _place_dot(inp, r2, c1, color_c)
        _place_relay_guides(inp, "row", r2, c1, side3, color_c, step3)
        return c1, r1

    raise ValueError("Failed to build row-based chain")


def _build_col_chain(
    inp: np.ndarray,
    active_cols: list[int],
    palette: list[int],
) -> tuple[int, int]:
    H, W = inp.shape
    free_cols = [c for c in range(1, W - 1) if c not in active_cols]
    if len(free_cols) < 1:
        raise ValueError("Need a free col for the final relay")

    colors = random.sample(palette, 3)
    color_a, color_b, color_c = map(int, colors)
    c1 = random.choice(active_cols)

    for _ in range(500):
        side1 = random.choice(["top", "bottom"])
        dir1 = 1 if side1 == "top" else -1
        max_step1 = min(_STEP_RANGE[1], H - 3)
        step1 = _pick_step(max_step1)
        source1 = 0 if dir1 > 0 else H - 1
        far1 = source1 + dir1 * (step1 + 1)
        companion_rows = [
            source1 + dir1 * k * step1
            for k in range(2, H)
            if 1 <= source1 + dir1 * k * step1 <= H - 2
            and dir1 * (source1 + dir1 * k * step1 - far1) > 0
        ]
        if not companion_rows:
            continue
        r1 = random.choice(companion_rows)

        side2 = random.choice(["left", "right"])
        dir2 = 1 if side2 == "left" else -1
        max_step2 = min(_STEP_RANGE[1], W - 3)
        step2 = _pick_step(max_step2)
        guide1 = c1 + dir2
        guide2 = c1 + dir2 * (step2 + 1)
        final_cols = [
            c1 + dir2 * k * step2
            for k in range(2, W)
            if 1 <= c1 + dir2 * k * step2 <= W - 2
            and c1 + dir2 * k * step2 in free_cols
            and dir2 * (c1 + dir2 * k * step2 - guide2) > 0
        ]
        if not (
            1 <= guide1 <= W - 2
            and 1 <= guide2 <= W - 2
            and guide1 not in active_cols
            and guide2 not in active_cols
            and final_cols
        ):
            continue
        c2 = random.choice(final_cols)

        side3 = random.choice(["top", "bottom"])
        dir3 = 1 if side3 == "top" else -1
        max_step3 = min(_STEP_RANGE[1], H - 3)
        step3 = _pick_step(max_step3)
        guide3a = r1 + dir3
        guide3b = r1 + dir3 * (step3 + 1)
        if not (1 <= guide3a <= H - 2 and 1 <= guide3b <= H - 2):
            continue

        _place_boundary_guides(inp, "col", c1, side1, color_a, step1)
        _place_dot(inp, r1, c1, color_b)
        _place_relay_guides(inp, "row", r1, c1, side2, color_b, step2)
        _place_dot(inp, r1, c2, color_c)
        _place_relay_guides(inp, "col", c2, r1, side3, color_c, step3)
        return r1, c1

    raise ValueError("Failed to build col-based chain")


def _choose_active_count(total_internal: int) -> int:
    max_active = min(_ACTIVE_LINES_RANGE[1], (total_internal + 1) // 2)
    if max_active < _ACTIVE_LINES_RANGE[0]:
        raise ValueError("Grid too small for the requested active-line range")
    return random.randint(_ACTIVE_LINES_RANGE[0], max_active)


def _gen_attempt() -> np.ndarray | None:
    by_row = random.choice([True, False])
    palette = random.sample(_COLORS, random.randint(5, min(8, len(_COLORS))))

    if by_row:
        H = random.randint(max(10, _ROWS_RANGE[0]), _ROWS_RANGE[1])
        W = random.randint(max(12, _COLS_RANGE[0]), _COLS_RANGE[1])
        try:
            n_active = _choose_active_count(H - 2)
        except ValueError:
            return None
        active0 = _spaced_indices(n_active, H - 2)
        if not active0:
            return None
        active_rows = [r + 1 for r in active0]
        inp = np.zeros((H, W), dtype=int)
        try:
            relay_col, _ = _build_row_chain(inp, active_rows, palette)
        except ValueError:
            return None

        side_a, side_b = "left", "right"
        remaining_rows = [r for r in active_rows if np.all(inp[r] == 0)]
        if len(remaining_rows) < 2:
            return None
        forced_sides = [side_a, side_b]
        random.shuffle(forced_sides)
        for idx, row in enumerate(remaining_rows):
            side = forced_sides[idx] if idx < 2 else random.choice([side_a, side_b])
            try:
                _build_simple_initial_line(
                    inp,
                    "row",
                    row,
                    side,
                    palette,
                    {relay_col},
                    require_companion=(idx == 2),
                )
            except ValueError:
                return None
    else:
        H = random.randint(max(10, _ROWS_RANGE[0]), _ROWS_RANGE[1])
        W = random.randint(max(12, _COLS_RANGE[0]), _COLS_RANGE[1])
        try:
            n_active = _choose_active_count(W - 2)
        except ValueError:
            return None
        active0 = _spaced_indices(n_active, W - 2)
        if not active0:
            return None
        active_cols = [c + 1 for c in active0]
        inp = np.zeros((H, W), dtype=int)
        try:
            relay_row, _ = _build_col_chain(inp, active_cols, palette)
        except ValueError:
            return None

        side_a, side_b = "top", "bottom"
        remaining_cols = [c for c in active_cols if np.all(inp[:, c] == 0)]
        if len(remaining_cols) < 2:
            return None
        forced_sides = [side_a, side_b]
        random.shuffle(forced_sides)
        for idx, col in enumerate(remaining_cols):
            side = forced_sides[idx] if idx < 2 else random.choice([side_a, side_b])
            try:
                _build_simple_initial_line(
                    inp,
                    "col",
                    col,
                    side,
                    palette,
                    {relay_row},
                    require_companion=(idx == 2),
                )
            except ValueError:
                return None

    try:
        out, meta = _simulate(inp)
    except Exception:
        return None
    if np.array_equal(inp, out):
        return None
    if int(meta["relay_count"]) < 2:
        return None
    if int(meta["initial_count"]) < _ACTIVE_LINES_RANGE[0]:
        return None
    if int(meta["initial_side_count"]) != 2:
        return None
    if by_row:
        if str(meta["initial_orientation"]) != "row":
            return None
    else:
        if str(meta["initial_orientation"]) != "col":
            return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
