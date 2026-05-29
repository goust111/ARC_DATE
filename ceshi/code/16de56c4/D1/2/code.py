# Variant of 16de56c4
# Dimension: D1 — 路径步长计算方式 (path)
# Change: step = 两个标记点之间的间距（即差值），offset = 第一个标记点位置 mod step... → 递增步长

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


# Design analysis for the D1 variant
# ----------------------------------
# a) Concrete behavior trace
#    The border marker still identifies the firing edge and the second marker
#    still fixes an initial gap.  The change is the path geometry: after the
#    first hop of length gap, the next hop is gap+1, then gap+2, etc.  Example
#    on a left-firing row of length 12:
#      markers at positions 0 and 2 -> gap = 2
#      old path: 0,2,4,6,8,10
#      new path: 0,2,5,9
#    If a companion sits on 9, the output paints 0,2,5,9 in the companion
#    color; if a bystander sits on 6, it is untouched because 6 is no longer on
#    the accelerated path.  This affects both solver alignment logic and how the
#    generator chooses legal companion/bystander positions.
# b) Complexity calibration
#    Keeping exactly two same-color markers per active line preserves the core
#    inference scaffold from the source task, while the irregular gaps create a
#    visible difference.  To avoid degenerate samples that look almost periodic,
#    the generator prefers initial gaps that yield at least four trajectory
#    cells whenever the line length allows it.
# c) Old implicit assumptions that break
#    - `offset = a % step` and `p % step == offset` assumed constant spacing.
#      This is invalid now and must be replaced by explicit trajectory
#      enumeration.
#    - Companion detection by modular class also breaks for the same reason.
#    - The original loop over `[lo, hi]` with a modular filter only worked
#      because every valid fill cell inside the span belonged to one residue
#      class.  The new variant must instead paint a prefix of the enumerated
#      trajectory.
# d) Placement legality
#    The generator constructs markers from the edge inward so the first two
#    trajectory positions are always occupied by the marker color.  A companion,
#    when present, is chosen from later trajectory positions only.  A bystander
#    is chosen from positions outside the trajectory, preventing overlap and
#    preventing accidental promotion to a companion.
# e) Visual observability
#    The irregular gaps are visible because outputs show spacing like 1,2,3,...
#    or 2,3,4,... between painted cells.  Samples where the path would stop
#    after only two cells are discouraged, since those would be visually too
#    close to the original constant-step task.
# f) Visual cleanliness
#    Each active line still contains only one marker pair plus at most one extra
#    dot, so lines remain sparse and readable.  Active lines are separated by at
#    least one empty line exactly as in the source task.
# g) Color inference
#    Colors still have the same semantics as the original: marker color paints
#    the path unless a single aligned companion overrides it.  No invisible
#    ordering by color id is introduced.
# h) Variant-specific generator metric
#    A useful sanity metric is `len(trajectory) - 2`, the number of cells beyond
#    the initial marker pair.  Values >= 2 mean the accelerated behavior is
#    actually visible; the generator is biased toward such lines.
# i) New solver assumptions to mirror in generation
#    The solver now assumes the repeated color includes the actual edge cell and
#    that any companion lies on a later trajectory cell.  The generator enforces
#    exactly that construction on every active line.


def _trajectory_positions(start: int, gap: int, direction: int, line_len: int) -> list[int]:
    """Enumerate the accelerated path: gap, gap+1, gap+2, ..."""
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
    """Return (edge_pos, inward_direction) for a shooting edge."""
    if edge in ("left", "top"):
        return 0, 1
    return line_len - 1, -1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill accelerating rays that launch from a unique grid edge.

    Input structure:
    - bg is the most frequent color; all other cells are isolated dots.
    - One boundary has strictly more non-bg dots than the other three and
      defines the shooting edge.
    - On each active shooting-direction line, exactly one color appears twice:
      one dot is on the shooting edge and the other is the first interior hit.
      Their distance gives the initial gap of the ray.
    - An optional third dot on the same line is either a companion placed on a
      later ray hit or a bystander placed off the ray.

    Rule:
    - Identify the unique shooting edge; left/right act across rows, top/bottom
      act across columns.
    - For a marker pair, compute the initial gap and enumerate the ray cells by
      taking jumps of gap, gap+1, gap+2, ... inward from the boundary marker.
    - A different-colored dot that lies on a later ray hit is a companion: it
      absorbs the ray, sets the paint color, and truncates filling at its hit.
    - A dot off the accelerating path is a bystander and remains untouched.
    - Without a companion, the marker color fills every hit of the accelerated
      path across the whole line.

    Key insight:
    The two repeated dots no longer define a constant modular beat.  They define
    only the first hop, and the remaining fill cells must be generated by
    explicitly walking an arithmetic-growth path from the shooting edge.
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
    winners = [e for e, c in edge_counts.items() if c == max_count]
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

        repeating = {c: ps for c, ps in colors.items() if len(ps) >= 2}
        assert repeating, (
            f"Line {li} ({'row' if by_row else 'col'}) has non-bg dots "
            f"{dict(colors)} but no repeating color"
        )

        for marker, mpos in repeating.items():
            assert len(mpos) == 2, (
                f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
                f"has {len(mpos)} dots (expected 2): {mpos}"
            )
            assert edge_pos in mpos, (
                f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
                f"must include shooting-edge position {edge_pos}: {mpos}"
            )

            gap = abs(mpos[0] - mpos[1])
            path = _trajectory_positions(edge_pos, gap, direction, line_len)
            assert len(path) >= 2, f"Line {li} gap {gap} yields no interior marker"
            assert path[1] in mpos, (
                f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
                f"does not match the first accelerated hop: {mpos}"
            )

            pos_to_idx = {p: idx for idx, p in enumerate(path)}
            companion_color = None
            companion_idx = None
            for oc, ops in colors.items():
                if oc == marker:
                    continue
                for p in ops:
                    idx = pos_to_idx.get(p)
                    if idx is not None and idx >= 2:
                        assert companion_color is None, (
                            f"Line {li} ({'row' if by_row else 'col'}) has "
                            f"multiple companions on the accelerated path"
                        )
                        companion_color = oc
                        companion_idx = idx

            if companion_color is None:
                paint = marker
                fill_positions = path
            else:
                paint = companion_color
                fill_positions = path[: companion_idx + 1]

            for p in fill_positions:
                rc = cell(li, p)
                cur = int(out[rc])
                if cur == bg or cur == marker:
                    out[rc] = paint

    return out


# --------------- generator ---------------

_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_GAP_RANGE = (1, 4)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_P_MARKER_ONLY = 0.40
_P_COMPANION = 0.35

_MAX_RETRIES = 400


def _choose_dims(edge: str, n_active: int):
    """Pick (H, W, num_lines, line_len) compatible with edge and active count."""
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
    """Choose n indices from [0, total) with gaps >= 1 between neighbors."""
    available = total - (n - 1)
    if available < n:
        return []
    picks = sorted(random.sample(range(available), n))
    return [picks[i] + i for i in range(n)]


def _candidate_gaps(line_len: int, min_path_len: int) -> list[int]:
    """Initial gaps whose accelerated path has at least min_path_len cells."""
    edge_pos, direction = 0, 1
    gaps = []
    for gap in range(_GAP_RANGE[0], min(_GAP_RANGE[1], line_len - 1) + 1):
        path = _trajectory_positions(edge_pos, gap, direction, line_len)
        if len(path) >= min_path_len:
            gaps.append(gap)
    return gaps


def _pick_gap(line_len: int) -> int | None:
    """Prefer gaps that make the non-periodic behavior visibly extend."""
    for min_path_len in (4, 3, 2):
        gaps = _candidate_gaps(line_len, min_path_len)
        if gaps:
            return random.choice(gaps)
    return None


def _build_line_dots(edge: str, line_len: int, marker_color: int, gap: int):
    """Build one active line using the accelerated-path semantics."""
    edge_pos, direction = _edge_params(edge, line_len)
    path = _trajectory_positions(edge_pos, gap, direction, line_len)
    assert len(path) >= 2
    inner_pos = path[1]

    dots = [(edge_pos, marker_color), (inner_pos, marker_color)]
    other_colors = [c for c in _COLORS if c != marker_color]

    roll = random.random()
    if roll < _P_MARKER_ONLY:
        pass
    elif roll < _P_MARKER_ONLY + _P_COMPANION:
        companion_positions = path[2:]
        if other_colors and companion_positions:
            dots.append((random.choice(companion_positions), random.choice(other_colors)))
    else:
        path_set = set(path)
        bystander_positions = [p for p in range(line_len) if p not in path_set]
        if other_colors and bystander_positions:
            dots.append((random.choice(bystander_positions), random.choice(other_colors)))

    return dots, edge_pos, path


def _count_other_edge_dots(
    dots_rc: list[tuple[int, int]], edge: str, H: int, W: int
) -> dict[str, int]:
    """Count how many placed dots land on each non-shooting edge."""
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


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input."""
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
    visible_variant_lines = 0

    for li in active_lines:
        marker_color = random.choice(_COLORS)
        gap = _pick_gap(line_len)
        if gap is None:
            return None

        dots, edge_pos, path = _build_line_dots(edge, line_len, marker_color, gap)
        visible_variant_lines += int(len(path) >= 4)

        for pos, color in dots:
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

        if any(pos == edge_pos for pos, _ in dots):
            shooting_edge_dots += 1

    if visible_variant_lines == 0:
        return None

    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(cnt >= shooting_edge_dots for cnt in other_counts.values()):
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
