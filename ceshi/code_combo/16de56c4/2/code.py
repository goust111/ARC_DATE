# Combo variant of 16de56c4 (index 2)
# combo_spec: D1/2+D5/0
# Name: 递增步长+碰撞停止

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np

try:
    from arc_utils import get_bg
except ImportError:
    sys.path.append(str(Path("/home/lixinyu/snap/new/solvers")))
    from arc_utils import get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace
#    Keep the source task scaffold: one unique shooting edge, sparse dots, and
#    on each active row/column exactly one color repeated twice. The repeated
#    pair still identifies a ray, but two old mechanisms change together:
#    - D1/2 replaces constant-period beats with an accelerated trajectory whose
#      jump lengths are gap, gap+1, gap+2, ...
#    - D5/0 removes companion recoloring/span logic and instead treats the
#      first non-background dot farther inward as a hard obstacle.
#    Example on a left-firing row of length 12:
#      markers at 0 and 2 -> initial gap = 2
#      source task would fill 0,2,4,6,8,10 or recolor a modular span
#      D1-only would fill 0,2,5,9
#      combo with blocker at 6 fills 0,2,5 and stops before the next hit at 9
#      combo with blocker at 5 fills only 0,2 because the collision cell itself
#      is not repainted
#    The visible signature is therefore uneven spacing plus line-specific early
#    truncation caused by both on-path and off-path blockers.
# b) Complexity calibration
#    To keep the rule inferable, each active line still has only one repeated
#    marker pair plus at most one extra dot. The generator deliberately mixes
#    three line types:
#    - clear accelerated rays
#    - blockers placed on a later accelerated hit
#    - blockers placed off the accelerated hits but still before a later hit
#    This is enough to show both dimensions without introducing several object
#    classes or ambiguous matching. Initial gaps are biased toward trajectories
#    of length >= 4 so the acceleration is actually visible.
# c) Old implicit assumptions that break
#    - Modular arithmetic (`p % step == offset`) no longer describes the fill;
#      the solver must explicitly enumerate the accelerated path.
#    - A differently colored dot no longer donates color, and there is no span
#      bounded by marker+companion. Output color always remains the marker color.
#    - An off-path dot is no longer irrelevant: if it lies ahead of the inner
#      marker, it still blocks the ray before later hits are painted.
#    - The fill loop must exclude the blocker cell itself, because collisions
#      stop before the obstacle rather than overwrite it.
# d) Placement legality
#    The generator constructs every marker pair from the shooting edge inward,
#    so the repeated color always occupies the first two accelerated hits.
#    A blocker, when present, is placed strictly beyond the inner marker in the
#    shooting direction, never overlaps a marker, and is chosen so that it
#    changes the output. Active lines remain separated by at least one empty
#    line/column, preserving the original visual spacing.
# e) Visual observability
#    Across a few training examples, a human can compare:
#    - clear lines that continue with growing gaps,
#    - lines stopped by a later on-path obstacle,
#    - lines stopped by an off-path obstacle that still lies earlier than the
#      next accelerated hit.
#    The off-path blocker cases are especially important because they prove the
#    stop rule depends on physical order along the line, not just membership in
#    the accelerated hit set.
# f) Visual cleanliness
#    There is at most one blocker per active line, no overlapping objects, and
#    the palette per grid is kept small. Output remains a sparse set of dots
#    rather than dense spans, so the accelerating pattern stays readable.
# g) New solver assumptions mirrored by generation
#    The solver assumes exactly one repeated color per active line, that this
#    color includes the shooting-edge cell, and that any extra dot lies beyond
#    the inner marker. The generator enforces these constraints and validates
#    the unique-edge condition before accepting a sample.


def _trajectory_positions(start: int, gap: int, direction: int, line_len: int) -> list[int]:
    """Enumerate the accelerated ray hits: gap, gap+1, gap+2, ..."""
    assert gap >= 1
    assert direction in (-1, 1)
    pos = start
    step = gap
    path = [start]
    while True:
        pos += direction * step
        if not (0 <= pos < line_len):
            break
        path.append(pos)
        step += 1
    return path


def _edge_params(edge: str, line_len: int) -> tuple[int, int]:
    """Return the shooting-edge position and inward direction."""
    if edge in ("left", "top"):
        return 0, 1
    return line_len - 1, -1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill accelerating rays that stop at the first visible collision.

    Input structure:
    - Background is the most common color; the rest are sparse dots.
    - Exactly one boundary has strictly more non-background dots than the other
      three; that boundary is the shooting edge.
    - On each active shooting-direction line, exactly one color appears twice:
      one marker sits on the shooting edge and the other is the first inward hit.
      Their distance gives the initial gap of the ray.
    - A line may also contain one extra non-background dot farther inward. It
      can lie on a later accelerated hit or between two hits.

    Rule:
    - Identify the unique shooting edge; left/right fire along rows, top/bottom
      fire along columns.
    - For each repeated marker pair, enumerate the ray cells by stepping inward
      with jump lengths gap, gap+1, gap+2, ...
    - Ignore color/alignment when searching for a stopper: the first non-bg dot
      beyond the inner marker is a solid obstacle.
    - Paint the marker color onto accelerated hit cells strictly before that
      obstacle. If no obstacle exists, paint every accelerated hit to the far side.
      The obstacle cell itself remains unchanged.

    Key insight:
    The repeated pair no longer defines a modular beat and the third dot no
    longer defines a recolored span. Solving the task requires explicit path
    enumeration plus a separate nearest-obstacle test along the firing direction.
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
    edge_pos, direction = _edge_params(edge, line_len)

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    for li in range(num_lines):
        colors: dict[int, list[int]] = defaultdict(list)
        for p in range(line_len):
            v = int(g[cell(li, p)])
            if v != bg:
                colors[v].append(p)

        if not colors:
            continue

        repeating = {color: ps for color, ps in colors.items() if len(ps) >= 2}
        assert len(repeating) == 1, (
            f"Line {li} ({'row' if by_row else 'col'}) must have exactly one "
            f"repeating marker color: {dict(colors)}"
        )
        marker, mpos = next(iter(repeating.items()))
        assert len(mpos) == 2, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} has "
            f"{len(mpos)} dots (expected 2): {mpos}"
        )
        assert edge_pos in mpos, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} must "
            f"include shooting-edge position {edge_pos}: {mpos}"
        )

        gap = abs(mpos[0] - mpos[1])
        path = _trajectory_positions(edge_pos, gap, direction, line_len)
        assert len(path) >= 2, f"Line {li} gap {gap} yields no interior hit"
        assert path[1] in mpos, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} does "
            f"not match the first accelerated hop: {mpos}"
        )
        inner_pos = path[1]

        blocker_positions: list[int] = []
        for color, ps in colors.items():
            if color == marker:
                continue
            for p in ps:
                if direction == 1:
                    assert p > inner_pos, (
                        f"Line {li} blocker must be beyond inner marker: "
                        f"marker={mpos}, blocker={p}"
                    )
                else:
                    assert p < inner_pos, (
                        f"Line {li} blocker must be beyond inner marker: "
                        f"marker={mpos}, blocker={p}"
                    )
                blocker_positions.append(p)

        if blocker_positions:
            stop_pos = min(blocker_positions) if direction == 1 else max(blocker_positions)
        else:
            stop_pos = line_len if direction == 1 else -1

        for p in path:
            if (direction == 1 and p >= stop_pos) or (direction == -1 and p <= stop_pos):
                break
            rc = cell(li, p)
            cur = int(out[rc])
            if cur == bg or cur == marker:
                out[rc] = marker

    return out


_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_GAP_RANGE = (1, 5)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")
_LINE_TYPES = ("clear", "aligned_blocker", "offpath_blocker")
_MAX_RETRIES = 500


def _choose_dims(edge: str, n_active: int) -> tuple[int, int, int, int] | None:
    """Pick (H, W, num_lines, line_len) compatible with the active-line count."""
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
    """Choose n indices from [0, total) with gaps >= 1."""
    available = total - (n - 1)
    if available < n:
        return []
    picks = sorted(random.sample(range(available), n))
    return [picks[i] + i for i in range(n)]


def _candidate_gaps(line_len: int, min_path_len: int) -> list[int]:
    """Initial gaps whose accelerated path has at least min_path_len cells."""
    gaps = []
    for gap in range(_GAP_RANGE[0], min(_GAP_RANGE[1], line_len - 1) + 1):
        path = _trajectory_positions(0, gap, 1, line_len)
        if len(path) >= min_path_len:
            gaps.append(gap)
    return gaps


def _pick_gap(line_len: int, line_type: str) -> int | None:
    """Bias toward visibly accelerated paths and blocker-sensitive geometry."""
    min_lengths = (4, 3, 2) if line_type == "clear" else (4, 3)
    for min_path_len in min_lengths:
        gaps = _candidate_gaps(line_len, min_path_len)
        if gaps:
            return random.choice(gaps)
    return None


def _aligned_blocker_positions(path: list[int], line_len: int) -> list[int]:
    """Later accelerated hits that can act as blockers."""
    candidates = path[2:]
    if not candidates:
        return []
    trimmed = [p for p in candidates if p not in (0, line_len - 1)]
    if trimmed:
        candidates = trimmed
    if len(candidates) >= 2:
        return candidates[1:] + candidates[:1]
    return candidates


def _offpath_blocker_positions(path: list[int], direction: int) -> list[int]:
    """Cells off the accelerated path that still lie before a later hit."""
    if len(path) < 3:
        return []

    preferred: list[int] = []
    fallback: list[int] = []
    path_set = set(path)

    intervals: list[tuple[int, int, bool]] = []
    if len(path) >= 4:
        intervals.append((path[2], path[3], True))
    intervals.append((path[1], path[2], False))
    if len(path) >= 5:
        for idx in range(3, len(path) - 1):
            intervals.append((path[idx], path[idx + 1], False))

    for start, end, is_preferred in intervals:
        lo, hi = sorted((start, end))
        cells = [p for p in range(lo + 1, hi) if p not in path_set]
        if direction == -1:
            cells = list(reversed(cells))
        if is_preferred:
            preferred.extend(cells)
        else:
            fallback.extend(cells)

    return preferred + fallback


def _build_line_dots(
    edge: str,
    line_len: int,
    marker_color: int,
    gap: int,
    blocker_color: int,
    line_type: str,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Construct one active line and return its dots plus the full path."""
    edge_pos, direction = _edge_params(edge, line_len)
    path = _trajectory_positions(edge_pos, gap, direction, line_len)
    assert len(path) >= 2

    dots = [(edge_pos, marker_color), (path[1], marker_color)]
    if line_type == "clear":
        return dots, path

    if line_type == "aligned_blocker":
        candidates = _aligned_blocker_positions(path, line_len)
    else:
        candidates = _offpath_blocker_positions(path, direction)
    if not candidates:
        raise AssertionError(f"No blocker candidates for {line_type} with path {path}")

    dots.append((random.choice(candidates), blocker_color))
    return dots, path


def _count_other_edge_dots(
    dots_rc: list[tuple[int, int]],
    edge: str,
    H: int,
    W: int,
) -> dict[str, int]:
    """Count how many placed dots land on each non-shooting boundary."""
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


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid combo input."""
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

    line_types = ["clear", "aligned_blocker", "offpath_blocker"]
    while len(line_types) < n_active:
        line_types.append(random.choice(_LINE_TYPES))
    random.shuffle(line_types)

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    shooting_edge_dots = 0
    all_rc: list[tuple[int, int]] = []
    visible_acceleration_lines = 0
    blocker_lines = 0
    offpath_lines = 0
    changed_by_collision = 0

    for li, line_type in zip(active_lines, line_types):
        gap = _pick_gap(line_len, line_type)
        if gap is None:
            return None

        marker_color = random.choice(palette)
        blocker_options = [c for c in palette if c != marker_color]
        if not blocker_options:
            blocker_options = [c for c in _COLORS if c != marker_color]
        if not blocker_options:
            return None
        blocker_color = random.choice(blocker_options)

        try:
            dots, path = _build_line_dots(
                edge=edge,
                line_len=line_len,
                marker_color=marker_color,
                gap=gap,
                blocker_color=blocker_color,
                line_type=line_type,
            )
        except AssertionError:
            return None

        path_set = set(path)
        visible_acceleration_lines += int(len(path) >= 4)
        blocker_pos = None if line_type == "clear" else dots[-1][0]
        if blocker_pos is not None:
            blocker_lines += 1
            offpath_lines += int(blocker_pos not in path_set)
            if (path[-1] > blocker_pos > path[1]) or (path[-1] < blocker_pos < path[1]):
                changed_by_collision += 1

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

    if visible_acceleration_lines == 0 or blocker_lines == 0 or offpath_lines == 0:
        return None
    if changed_by_collision < 2:
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
