# Variant of 16de56c4
# Dimension: D3 — 轨迹填充模式 (trajectory)
# Change: 等间距连续填充：在beat位置(pos % step == offset)全部填充相同颜色... → 颜色循环

from __future__ import annotations

from collections import defaultdict
import random
import sys

import numpy as np

sys.path.append("/home/lixinyu/snap/new/solvers")
from arc_utils import get_bg


# Design analysis for the D3 variant
# a) Concrete behavior
#    - The old task used one paint color per active line: either the marker color
#      on the whole line, or the aligned companion color inside a bounded span.
#    - The variant keeps the same geometry: one shooting edge, one repeated marker
#      color per active line, optional aligned companion, optional unaligned
#      bystander.
#    - The only changed stage is the beat-filling stage. Once step/range are known,
#      the beat cells are no longer painted uniformly. Instead, the non-bg dots on
#      that line are read in trajectory order (from the shooting edge inward) to
#      form a cycle such as [marker, marker, other]. Beat cells consume that cycle
#      repeatedly in the same trajectory order.
#    - Example, left edge row with marker dots at positions 0 and 2, bystander at 5:
#      old output on beats 0,2,4,6,8 was all marker color; new output becomes
#      marker, marker, bystander, marker, marker, ...
#    - Example, top edge column with marker dots at rows 0 and 3, companion at 6:
#      old bounded fill on rows 0,3,6 was all companion color; new bounded fill is
#      marker, marker, companion.
# b) Complexity calibration
#    - The cycle is deliberately short: 1 color for marker-only lines, 3 entries
#      for lines with an extra dot. This is more visible than the original rule but
#      still simple enough to infer from 2-4 examples.
#    - The generator forces at least one bystander line and one companion line so
#      both "unbounded cycle" and "bounded cycle" appear repeatedly across samples.
# c) Implicit-assumption audit of the original solver
#    - The original code assumed a single scalar `paint` per line. That is the main
#      invalid assumption under this variant.
#    - The old write condition `if cur == bg or cur == marker` is still valid and is
#      preserved: it lets companions/bystanders remain visible as source clues while
#      marker cells can be recolored by the cycle.
#    - The original "first aligned non-marker dot is the companion" assumption still
#      holds because the generator continues to build at most one aligned extra dot.
# d) Placement legality
#    - Active lines remain distinct and spaced apart, so no two line objects overlap.
#    - Every line still contains exactly one repeated marker color with exactly two
#      dots; extra dots are singletons of different colors.
#    - Bystanders are placed beyond the inner marker and off-beat. Companions are
#      placed on the immediate next beat beyond the inner marker so the bounded
#      three-beat pattern is always well-defined and easy to read.
# e) Visual observability
#    - The cycle order depends only on visible spatial order from the deducible
#      shooting edge; no color-id sorting or hidden metadata is used.
#    - Requiring at least one bystander line prevents the generator from collapsing
#      back to a purely uniform-fill dataset.
# f) Visual cleanliness
#    - Each active line remains sparse in the input and 1D in the output, so the
#      added color variation does not create overlapping clutter.
# g) Color-rule inferability
#    - Distinct colors per grid are limited by sampling from a small palette subset.
#      The same colors repeat across lines, which helps humans identify that the
#      output sequence is copied from the input dots rather than chosen arbitrarily.
# h) Variant anti-degeneration check
#    - The generator compares the new solver output against a local reimplementation
#      of the legacy rule and rejects any sample where they are identical.
# i) New solver assumptions and generator guarantees
#    - `main()` now assumes every multi-color line exposes its cycle in trajectory
#      order. The generator guarantees that by placing extras only beyond the inner
#      marker, so the first two cycle entries are always the two marker dots.


def _detect_edge(g: np.ndarray, bg: int) -> str:
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
    return winners[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill periodic beat positions launched from a unique grid edge, but cycle colors along each trajectory.

    Input structure:
    - bg is the most frequent color; non-bg content is sparse and lies on a small
      set of rows or columns shot from one edge.
    - Each active shooting-direction line has exactly one repeated marker color
      with exactly two dots; their gap defines the beat step.
    - A line may also contain one extra singleton dot of a different color:
      aligned singletons are companions, unaligned singletons are bystanders.

    Rule:
    - Identify the unique shooting edge; left/right shoot across rows, top/bottom
      shoot across columns.
    - On each active line, use the two marker dots to recover `step` and the beat
      class. An aligned extra dot, if present, bounds the fill span exactly as in
      the original task; otherwise the whole line is fillable.
    - Read the non-bg dots on that line in trajectory order, from the shooting
      edge inward, and use their colors as a repeating cycle.
    - Paint beat cells in the same trajectory order by consuming that cycle
      repeatedly. Companions and bystanders remain as visible source dots; marker
      cells may be recolored by the cycle.

    Key insight:
    - The geometry still comes from the unique edge and the two marker dots, but
      the fill color is no longer a single line-wide constant. The visible dot
      sequence on each trajectory supplies a local color cycle that must be reused
      beat-by-beat.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    H, W = g.shape

    edge = _detect_edge(g, bg)
    by_row = edge in ("left", "right")
    num_lines = H if by_row else W
    line_len = W if by_row else H
    forward = edge in ("left", "top")

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    for li in range(num_lines):
        dots: list[tuple[int, int]] = []
        colors: dict[int, list[int]] = defaultdict(list)
        for p in range(line_len):
            v = int(g[cell(li, p)])
            if v != bg:
                dots.append((p, v))
                colors[v].append(p)

        if not dots:
            continue

        repeating = [(color, positions) for color, positions in colors.items() if len(positions) >= 2]
        assert len(repeating) == 1, (
            f"Line {li} ({'row' if by_row else 'col'}) must have exactly one repeating color: {dict(colors)}"
        )
        marker, mpos = repeating[0]
        assert len(mpos) == 2, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
            f"has {len(mpos)} dots (expected 2): {mpos}"
        )

        a, b = sorted(mpos)
        step = b - a
        offset = a % step

        aligned = [
            (p, c)
            for p, c in dots
            if c != marker and p % step == offset
        ]
        assert len(aligned) <= 1, (
            f"Line {li} ({'row' if by_row else 'col'}) has multiple aligned companions: {aligned}"
        )

        if aligned:
            comp_pos, _ = aligned[0]
            lo, hi = min(a, b, comp_pos), max(a, b, comp_pos)
        else:
            lo, hi = 0, line_len - 1

        ordered_dots = sorted(dots, key=lambda item: item[0], reverse=not forward)
        cycle = [color for _, color in ordered_dots]

        beat_positions = [p for p in range(lo, hi + 1) if p % step == offset]
        if not forward:
            beat_positions.reverse()

        for idx, p in enumerate(beat_positions):
            paint = int(cycle[idx % len(cycle)])
            rc = cell(li, p)
            cur = int(out[rc])
            if cur == bg or cur == marker:
                out[rc] = paint

    return out


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    """Original uniform-fill rule, kept only for generator anti-degeneration checks."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    H, W = g.shape

    edge = _detect_edge(g, bg)
    by_row = edge in ("left", "right")
    num_lines = H if by_row else W
    line_len = W if by_row else H

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

        repeating = [(color, positions) for color, positions in colors.items() if len(positions) >= 2]
        if len(repeating) != 1:
            raise AssertionError(f"Legacy check expected one repeating color on line {li}: {dict(colors)}")
        marker, mpos = repeating[0]
        if len(mpos) != 2:
            raise AssertionError(f"Legacy check expected exactly two marker dots on line {li}: {mpos}")

        a, b = sorted(mpos)
        step = b - a
        offset = a % step

        companion = None
        for color, positions in colors.items():
            if color == marker:
                continue
            if any(p % step == offset for p in positions):
                companion = color
                break

        if companion is not None:
            paint = companion
            all_ps = mpos + colors[companion]
            lo, hi = min(all_ps), max(all_ps)
        else:
            paint = marker
            lo, hi = 0, line_len - 1

        for p in range(lo, hi + 1):
            if p % step != offset:
                continue
            rc = cell(li, p)
            cur = int(out[rc])
            if cur == bg or cur == marker:
                out[rc] = paint

    return out


# --------------- generator ---------------

_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")
_LINE_TYPE_WEIGHTS = {
    "marker_only": 0.20,
    "companion": 0.25,
    "bystander": 0.55,
}
_MAX_RETRIES = 400


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


def _valid_steps(line_len: int, line_type: str) -> list[int]:
    max_step = min(_STEP_RANGE[1], line_len - 1)
    steps = []
    for step in range(_STEP_RANGE[0], max_step + 1):
        if line_type == "marker_only":
            steps.append(step)
        elif line_type == "companion":
            if 2 * step < line_len:
                steps.append(step)
        elif line_type == "bystander":
            if step >= 2 and 2 * step < line_len:
                steps.append(step)
    return steps


def _choose_line_types(n_active: int) -> list[str]:
    types = ["bystander", "companion"]
    choices = list(_LINE_TYPE_WEIGHTS.keys())
    weights = [_LINE_TYPE_WEIGHTS[k] for k in choices]
    while len(types) < n_active:
        types.append(random.choices(choices, weights=weights, k=1)[0])
    random.shuffle(types)
    return types


def _build_line_dots(
    edge: str,
    line_len: int,
    marker_color: int,
    extra_color: int,
    line_type: str,
    step: int,
) -> list[tuple[int, int]]:
    if edge in ("left", "top"):
        edge_pos = 0
        inner_pos = step
        forward = 1
    else:
        edge_pos = line_len - 1
        inner_pos = line_len - 1 - step
        forward = -1

    offset = edge_pos % step
    dots = [(edge_pos, marker_color), (inner_pos, marker_color)]

    if line_type == "marker_only":
        return dots

    if line_type == "companion":
        comp_pos = inner_pos + forward * step
        dots.append((comp_pos, extra_color))
        return dots

    if forward == 1:
        candidates = [p for p in range(inner_pos + 1, line_len) if p % step != offset]
    else:
        candidates = [p for p in range(inner_pos - 1, -1, -1) if p % step != offset]
    assert candidates, f"No bystander candidate for line_len={line_len}, step={step}, edge={edge}"
    dots.append((random.choice(candidates), extra_color))
    return dots


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
    for other_edge in _EDGES:
        if other_edge == edge:
            continue
        check = boundary[other_edge]
        counts[other_edge] = sum(1 for r, c in dots_rc if check(r, c))
    return counts


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

    line_types = _choose_line_types(n_active)
    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []
    shooting_edge_dots = 0
    palette = random.sample(_COLORS, random.randint(3, min(6, len(_COLORS))))

    for li, line_type in zip(active_lines, line_types):
        steps = _valid_steps(line_len, line_type)
        if not steps:
            return None
        step = random.choice(steps)

        marker_color = random.choice(palette)
        extra_choices = [color for color in palette if color != marker_color]
        if not extra_choices:
            return None
        extra_color = random.choice(extra_choices)

        dots = _build_line_dots(edge, line_len, marker_color, extra_color, line_type, step)
        used_positions = set()
        for pos, color in dots:
            if pos in used_positions:
                return None
            used_positions.add(pos)
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
        legacy = _legacy_main(inp)
    except Exception:
        return None

    if np.array_equal(inp, out):
        return None
    if np.array_equal(out, legacy):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
