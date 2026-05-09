# Combo variant of 16de56c4 (index 4)
# combo_spec: D5/0+D17/0
# Name: 碰撞停止+旁观者阻挡

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    The original task had two meanings for the third dot on a shooting line:
#    - aligned extra dot -> companion that recolors the beat span
#    - off-beat extra dot -> inert bystander
#    This combo removes that split. After the two marker dots establish the
#    step, every later non-background dot on the same row/column is just a hard
#    collision point. The ray keeps the marker color and stops before the first
#    such point, no matter whether that point is on a beat.
#    Example with edge=left and step=2:
#    - aligned extra at 6:
#      input: 3 . 3 . . . 8 . . . .
#      old  : 8 . 8 . 8 . 8 . . . .   (companion recolors through its own cell)
#      new  : 3 . 3 . 3 . 8 . . . .   (same dot is now only a blocker)
#    - off-beat extra at 5:
#      input: 3 . 3 . . 8 . . . . .
#      old  : 3 . 3 . 3 8 3 . 3 . 3   (bystander ignored)
#      new  : 3 . 3 . 3 8 . . . . .   (bystander also blocks)
#    The fused behavior is therefore stricter than either original branch:
#    "all forward non-bg pixels terminate the ray" is the whole rule.
#
# b) Complexity calibration
#    The object vocabulary stays compact: each active line has one marker pair
#    and optionally one extra blocker. To keep the fusion visually evident, the
#    generator mixes exactly three line archetypes within a grid:
#    - clear lines with no blocker,
#    - aligned blockers (formerly companions),
#    - off-beat blockers (formerly bystanders).
#    This gives contrast without introducing multiple blocker classes or
#    ambiguous multi-event races on the same line.
#
# c) Hidden assumptions broken from the original solver
#    - "Aligned extra dot changes paint color" is no longer valid.
#    - "Off-beat extra dot can be ignored" is no longer valid.
#    - "Fill range can be derived from a marker/companion span" is no longer
#      valid; directional nearest-collision logic replaces span logic.
#    The new solver therefore only trusts: unique shooting edge, exactly one
#    repeated marker color per active line, and marker pair touching the edge.
#
# d) Placement legality handled by the generator
#    - Exactly one repeated color appears on each active line.
#    - The extra dot, if present, is strictly beyond the inner marker in the
#      firing direction, so the first two hits still define the step cleanly.
#    - There is at most one extra dot per line, avoiding overlap and removing
#      ambiguity about the first collision.
#    - Candidate blocker positions are chosen so blocker lines visibly differ
#      from the input/output of the original task rather than degenerating into
#      "no new painted cell".
#
# e) Visual observability
#    A human can compare several lines in one sample:
#    - clear lines propagate to the far side,
#    - aligned blockers stop the ray and keep their own color,
#    - off-beat blockers also stop the ray.
#    Because both aligned and off-beat third dots now have the same physical
#    effect, the intended inference is geometric, not color-identity based.
#
# f) Visual cleanliness
#    Active lines are spaced apart and each line carries at most three dots in
#    the input. The output adds a single periodic trace per line, so there is no
#    dense overlap or multi-layer clutter inside one local region.
#
# g) Color-change inferability
#    There is no longer any conditional recoloring rule to infer. Colors remain
#    local and literal: the marker color is always the paint color, while any
#    foreign dot simply persists as an obstacle. A small per-grid palette keeps
#    the scene readable.
#
# h) Post-change solver assumptions reviewed against the generator
#    - Solver assumption: the repeated color has exactly two dots.
#      Generator guarantee: every active line starts from one edge marker pair.
#    - Solver assumption: the marker touching the dominant edge is the first
#      beat, and the second marker is the next beat inward.
#      Generator guarantee: markers are placed at beats[0] and beats[1].
#    - Solver assumption: every non-marker dot on the line lies beyond the inner
#      marker, so it is a forward collision candidate.
#      Generator guarantee: aligned/off-beat candidates are sampled only there.
#    - Solver assumption: the dominant edge is unique.
#      Generator guarantee: boundary counts are checked before acceptance.


def _edge_setup(edge: str, line_len: int, step: int) -> tuple[int, int, list[int]]:
    edge_pos = 0 if edge in ("left", "top") else line_len - 1
    direction = 1 if edge in ("left", "top") else -1
    beats = (
        list(range(edge_pos, line_len, step))
        if direction == 1
        else list(range(edge_pos, -1, -step))
    )
    return edge_pos, direction, beats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend periodic rays from one uniquely populated edge until they collide.

    Input structure:
    - Background is the most common color and the remaining pixels are sparse.
    - Exactly one boundary has strictly more non-background dots than the other
      three; that boundary sets the firing direction.
    - Each active row/column contains one repeated marker color with exactly two
      dots: one on the shooting edge and one step cells inward.
    - A line may also contain extra non-background dots farther ahead; they may
      be aligned to the beat or off the beat.

    Rule:
    - Detect the unique shooting edge; left/right fire across rows and
      top/bottom fire across columns.
    - Use the marker pair on each active line to recover the step and ordered
      beat positions moving away from the edge.
    - Paint beat cells with the marker color, starting from the edge, until the
      ray would reach the first later non-background cell on that line.
    - That first extra cell is never recolored and always blocks further travel,
      whether it is beat-aligned or not. If no such cell exists, the beat fills
      continue to the far boundary.

    Key insight: once the step is inferred from the two edge-anchored markers,
    every other visible dot ahead is treated uniformly as a collision object, so
    the puzzle becomes about directional nearest-obstacle reasoning rather than
    companion-vs-bystander classification.
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
    winners = [name for name, count in edge_counts.items() if count == max_count]
    assert len(winners) == 1, f"Ambiguous shooting edge: {winners} (counts={edge_counts})"
    edge = winners[0]

    by_row = edge in ("left", "right")
    num_lines = H if by_row else W
    line_len = W if by_row else H

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    for line in range(num_lines):
        colors: dict[int, list[int]] = defaultdict(list)
        for pos in range(line_len):
            value = int(g[cell(line, pos)])
            if value != bg:
                colors[value].append(pos)

        if not colors:
            continue

        repeating = {color: positions for color, positions in colors.items() if len(positions) >= 2}
        assert len(repeating) == 1, (
            f"Line {line} ({'row' if by_row else 'col'}) must have exactly one "
            f"repeating marker color: {dict(colors)}"
        )
        marker, marker_positions = next(iter(repeating.items()))
        assert len(marker_positions) == 2, (
            f"Line {line} ({'row' if by_row else 'col'}) marker {marker} has "
            f"{len(marker_positions)} dots (expected 2): {marker_positions}"
        )

        a, b = sorted(marker_positions)
        step = b - a
        assert step > 0, f"Line {line} marker {marker} has zero step"

        edge_pos, direction, beats = _edge_setup(edge, line_len, step)
        assert edge_pos in marker_positions, (
            f"Line {line} ({'row' if by_row else 'col'}) marker {marker} "
            f"does not touch the shooting edge: {marker_positions}"
        )
        inner_pos = beats[1]
        assert set(marker_positions) == {edge_pos, inner_pos}, (
            f"Line {line} marker pair inconsistent with edge/step order: "
            f"edge={edge_pos}, beats[:2]={beats[:2]}, markers={marker_positions}"
        )

        blockers: list[int] = []
        for other_color, positions in colors.items():
            if other_color == marker:
                continue
            for pos in positions:
                assert (pos - inner_pos) * direction > 0, (
                    f"Line {line} non-marker dot must lie beyond inner marker: "
                    f"markers={marker_positions}, dot={pos}"
                )
                blockers.append(pos)

        stop_pos = None
        if blockers:
            stop_pos = min(blockers, key=lambda pos: (pos - edge_pos) * direction)

        for pos in beats:
            if stop_pos is not None and (pos - stop_pos) * direction >= 0:
                break
            rc = cell(line, pos)
            current = int(out[rc])
            if current == bg or current == marker:
                out[rc] = marker

    return out


_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")
_LINE_TYPES = ("clear", "aligned_blocker", "offset_blocker")
_MAX_RETRIES = 500


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


def _line_candidates(
    edge: str, line_len: int, step: int
) -> tuple[int, int, list[int], list[int], list[int]]:
    edge_pos, direction, beats = _edge_setup(edge, line_len, step)
    if len(beats) < 3:
        return edge_pos, direction, beats, [], []

    far_edge = line_len - 1 if direction == 1 else 0
    aligned = [pos for pos in beats[3:] if pos != far_edge]
    if len(beats) < 4:
        offset = []
    elif direction == 1:
        offset = [pos for pos in range(beats[2] + 1, beats[-1]) if pos % step != edge_pos % step]
    else:
        offset = [pos for pos in range(beats[-1] + 1, beats[2]) if pos % step != edge_pos % step]
    return edge_pos, direction, beats, aligned, offset


def _feasible_steps(edge: str, line_len: int, line_type: str) -> list[int]:
    max_step = min(_STEP_RANGE[1], line_len - 1)
    steps: list[int] = []
    for step in range(_STEP_RANGE[0], max_step + 1):
        _, _, beats, aligned, offset = _line_candidates(edge, line_len, step)
        if len(beats) < 3:
            continue
        if line_type == "clear":
            steps.append(step)
        elif line_type == "aligned_blocker" and aligned:
            steps.append(step)
        elif line_type == "offset_blocker" and offset:
            steps.append(step)
    return steps


def _sample_line_dots(
    edge: str,
    line_len: int,
    palette: list[int],
    line_type: str,
) -> list[tuple[int, int]] | None:
    steps = _feasible_steps(edge, line_len, line_type)
    if not steps:
        return None

    step = random.choice(steps)
    edge_pos, _, beats, aligned, offset = _line_candidates(edge, line_len, step)
    marker_color = random.choice(palette)
    dots = [(edge_pos, marker_color), (beats[1], marker_color)]

    if line_type == "clear":
        return dots

    other_colors = [color for color in palette if color != marker_color]
    if not other_colors:
        other_colors = [color for color in _COLORS if color != marker_color]
    if not other_colors:
        return None

    candidates = aligned if line_type == "aligned_blocker" else offset
    if not candidates:
        return None

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


def _choose_line_types(n_active: int) -> list[str]:
    line_types = ["clear", "aligned_blocker", "offset_blocker"]
    while len(line_types) < n_active:
        line_types.append(random.choice(_LINE_TYPES))
    random.shuffle(line_types)
    return line_types


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

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    shooting_edge_dots = 0
    all_rc: list[tuple[int, int]] = []
    aligned_lines = 0
    offset_lines = 0

    for line, line_type in zip(active_lines, _choose_line_types(n_active)):
        dots = _sample_line_dots(edge, line_len, palette, line_type)
        if dots is None:
            return None

        for pos, color in dots:
            r, c = cell(line, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

        shooting_edge_dots += 1
        if line_type == "aligned_blocker":
            aligned_lines += 1
        elif line_type == "offset_blocker":
            offset_lines += 1

    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(count >= shooting_edge_dots for count in other_counts.values()):
        return None
    if aligned_lines == 0 or offset_lines == 0:
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
