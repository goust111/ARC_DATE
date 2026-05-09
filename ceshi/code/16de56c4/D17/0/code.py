# Variant of 16de56c4
# Dimension: D17 — 旁观者点行为 (obstacle)
# Change: 旁观者点(不对齐beat的异色点)在输出中完全保持不变... → 旁观者作为障碍

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg

# Design analysis
# - Concrete behavior:
#   Each active row/column still has one repeated marker color with two dots:
#   one on the shooting edge and one interior, which sets the beat spacing.
#   A same-line aligned foreign dot is still a companion and recolors the ray.
#   The D17 change is only about an unaligned foreign dot: instead of being
#   inert, it is now a solid blocker. The ray travels away from the shooting
#   edge; once it reaches that off-beat dot, all later beat cells remain blank.
# - Trace on one row:
#   Old rule with edge=left, markers at positions 0 and 2, bystander at 5:
#   beats are 0,2,4,6,8,... and the row fills all of them. New rule: the ray
#   reaches 4, then hits the obstacle at 5, so 6,8,... stay background. The
#   obstacle itself keeps its own color because it is not repainted.
# - Complexity calibration:
#   The generator keeps only three line archetypes: marker-only, companion, and
#   obstacle. That is enough to show the new interaction without introducing
#   extra object vocabularies or ambiguous multi-dot lines.
# - Hidden assumption audit:
#   Edge detection and step inference are unchanged, so the generator still
#   guarantees a unique most-populated boundary and exactly two marker dots per
#   active line. The new solver additionally relies on blockers being beyond the
#   calibration pair; otherwise a blocker could sit between the two marker dots
#   and make the visual story incoherent. The generator enforces that.
# - Placement legality:
#   Obstacle lines require at least four beat positions. The blocker is sampled
#   from a non-beat cell strictly after the first newly fillable beat and before
#   a later beat, so it both allows some propagation and visibly stops more.
# - Visual observability:
#   Every generated grid contains at least one obstacle line that differs from
#   the original task: the old solver would fill past that foreign dot, while
#   the new solver leaves downstream beats empty. Companion and marker-only lines
#   remain present, giving side-by-side contrast inside the same puzzle.
# - Visual cleanliness / color inference:
#   Colors still act locally: marker color fills when unobstructed, companion
#   color fills only when the aligned dot is the first relevant event. There is
#   no hidden color ordering or global color mapping.
# - Post-change assumptions checked against the generator:
#   The solver looks for the nearest non-marker event beyond the inner marker on
#   each line. The generator keeps at most one non-marker dot per line, so this
#   nearest-event logic is unambiguous and easy for both code and humans.


def _edge_setup(edge: str, line_len: int, step: int) -> tuple[int, int, list[int]]:
    edge_pos = 0 if edge in ("left", "top") else line_len - 1
    direction = 1 if edge in ("left", "top") else -1
    if direction == 1:
        beats = list(range(edge_pos, line_len, step))
    else:
        beats = list(range(edge_pos, -1, -step))
    return edge_pos, direction, beats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate periodic rays from one dominant edge, with off-beat dots acting as blockers.

    Input structure:
    - Background is the most frequent color; the rest are sparse dots.
    - One boundary has strictly more non-background dots than the other three;
      that boundary is the shooting edge.
    - On each active shooting-direction line there is exactly one repeated
      marker color with two dots: one on the shooting edge and one farther in.
    - A different-colored aligned dot may appear farther along the same beat;
      a different-colored unaligned dot may also appear farther along the line.

    Rule:
    - Detect the unique shooting edge; left/right fire across rows, top/bottom
      fire across columns.
    - The two marker dots determine the step and therefore the ordered beat
      positions visited by the ray.
    - If the first relevant event beyond the inner marker is an aligned foreign
      dot, the ray fills beat cells up to that dot using the companion color.
    - If the first relevant event beyond the inner marker is an unaligned
      foreign dot, the ray stops before that obstacle and later beats stay blank.
      With no such event, the marker color fills the whole line.

    Key insight:
    - The task is still solved line-by-line after identifying the dominant
      boundary. The new ingredient is directional ordering: off-beat dots are no
      longer ignored, because the nearest one beyond the calibration pair caps
      how far the periodic beat sequence can continue.
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
            assert step > 0, f"Line {li} marker {marker} has zero step"
            edge_pos, direction, beats = _edge_setup(edge, line_len, step)
            assert edge_pos in mpos, (
                f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
                f"does not touch shooting edge: {mpos}"
            )
            inner_marker = b if a == edge_pos else a
            assert beats[:2] == [edge_pos, inner_marker], (
                f"Line {li} marker {marker} inconsistent with beat order: "
                f"edge={edge_pos}, inner={inner_marker}, beats={beats[:3]}"
            )
            offset = edge_pos % step

            companions: list[tuple[int, int, int]] = []
            obstacles: list[tuple[int, int]] = []
            for other_color, ops in colors.items():
                if other_color == marker:
                    continue
                for p in ops:
                    if (p - inner_marker) * direction <= 0:
                        continue
                    distance = (p - edge_pos) * direction
                    if p % step == offset:
                        companions.append((distance, p, other_color))
                    else:
                        obstacles.append((distance, p))

            companions.sort()
            obstacles.sort()
            nearest_companion = companions[0] if companions else None
            nearest_obstacle = obstacles[0] if obstacles else None

            paint = marker
            if nearest_companion is not None and (
                nearest_obstacle is None or nearest_companion[0] < nearest_obstacle[0]
            ):
                paint = nearest_companion[2]
                stop_distance = nearest_companion[0]
                fill_positions = [p for p in beats if (p - edge_pos) * direction <= stop_distance]
            elif nearest_obstacle is not None:
                stop_distance = nearest_obstacle[0]
                fill_positions = [p for p in beats if (p - edge_pos) * direction < stop_distance]
            else:
                fill_positions = beats

            for p in fill_positions:
                rc = cell(li, p)
                current = int(out[rc])
                if current == bg or current == marker:
                    out[rc] = paint

    return out


_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_LINE_TYPE_WEIGHTS = (
    ("marker", 0.30),
    ("companion", 0.30),
    ("obstacle", 0.40),
)

_MAX_RETRIES = 400


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


def _obstacle_candidates(beats: list[int], step: int, direction: int) -> list[int]:
    if len(beats) < 4:
        return []
    offset = beats[0] % step
    if direction == 1:
        return [p for p in range(beats[2] + 1, beats[-1]) if p % step != offset]
    return [p for p in range(beats[-1] + 1, beats[2]) if p % step != offset]


def _choose_step(edge: str, line_len: int, line_type: str) -> int | None:
    max_step = min(_STEP_RANGE[1], line_len - 1)
    steps = list(range(_STEP_RANGE[0], max_step + 1))
    random.shuffle(steps)
    for step in steps:
        edge_pos, direction, beats = _edge_setup(edge, line_len, step)
        del edge_pos
        if len(beats) < 3:
            continue
        if line_type == "obstacle" and not _obstacle_candidates(beats, step, direction):
            continue
        return step
    return None


def _build_line_dots(edge: str, line_len: int, marker_color: int, line_type: str):
    step = _choose_step(edge, line_len, line_type)
    if step is None:
        return None
    edge_pos, direction, beats = _edge_setup(edge, line_len, step)
    inner_pos = beats[1]
    dots = [(edge_pos, marker_color), (inner_pos, marker_color)]
    other_colors = [color for color in _COLORS if color != marker_color]

    if line_type == "companion":
        companion_pos = random.choice(beats[2:])
        dots.append((companion_pos, random.choice(other_colors)))
    elif line_type == "obstacle":
        obstacle_pos = random.choice(_obstacle_candidates(beats, step, direction))
        dots.append((obstacle_pos, random.choice(other_colors)))

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
    counts = {}
    for other_edge in _EDGES:
        if other_edge == edge:
            continue
        check = boundary[other_edge]
        counts[other_edge] = sum(1 for r, c in dots_rc if check(r, c))
    return counts


def _choose_line_types(n_active: int) -> list[str]:
    min_obstacles = max(1, n_active // 3)
    types = ["obstacle"] * min_obstacles
    labels = [label for label, _ in _LINE_TYPE_WEIGHTS]
    weights = [weight for _, weight in _LINE_TYPE_WEIGHTS]
    while len(types) < n_active:
        types.append(random.choices(labels, weights=weights, k=1)[0])
    random.shuffle(types)
    return types


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

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    shooting_edge_dots = 0
    all_rc: list[tuple[int, int]] = []

    for li, line_type in zip(active_lines, _choose_line_types(n_active)):
        marker_color = random.choice(_COLORS)
        dots = _build_line_dots(edge, line_len, marker_color, line_type)
        if dots is None:
            return None

        for pos, color in dots:
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

        shooting_edge_dots += 1

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
