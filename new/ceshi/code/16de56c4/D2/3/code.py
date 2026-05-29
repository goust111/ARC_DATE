# Variant of 16de56c4
# Dimension: D2 — 路径方向确定方式 (path)
# Change: 射击边由四条边上非背景点计数最多的边决定，left/right边对应行方向，top/bottom边对应列方向... → 双向射击

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D2 variant
# a) Concrete behavior:
#    The original task had one emitting edge, so each active row/column carried one
#    marker pair and one inward periodic fill.  Here an active line has two marker
#    pairs, one anchored to each opposite edge of the chosen axis.  Example on a row:
#    left marker color A at positions [0, 3], right marker color B at [8, 11].
#    Original logic would only keep one of those sides and fill one progression.
#    Variant logic lets both sides emit inward.  If both claim position 6, we compare
#    travel steps from the edge markers; the faster ray wins, and equal-time meetings
#    break toward the right/bottom side.  This is the core visible difference.
# b) Complexity calibration:
#    To keep the change visible, every non-empty line contains both edge markers.
#    At least one line per grid is forced to have a genuine overlap zone, so the
#    "meeting in the middle" rule is observable instead of degenerating into two
#    independent fills.  Extras stay limited to at most one per line to avoid clutter.
# c) Hidden assumptions from the old solver:
#    - Old code assumed exactly one winning edge; this now fails because left/right
#      or top/bottom deliberately tie.  We replace it with opposite-pair counting.
#    - Old code assumed one repeating color per line was enough to define the fill.
#      Now each active line needs one marker pair per opposite side.
#    - Old code could safely paint only bg / own-marker cells because no competing
#      ray existed.  Now both rays may target the same empty cells, so we must first
#      collect claims and then resolve them with an explicit priority rule.
# d) Placement legality:
#    The generator enforces two distinct marker colors per active line, one touching
#    each edge.  Inner markers never overlap, companions (when present) align to only
#    one source, and bystanders align to neither source.  This keeps solver parsing
#    unambiguous and preserves the intended line-local structure.
# e) Visual observability:
#    The task stays human-legible because the axis is still inferred from boundary
#    density, but now both opposite edges on that axis are populated.  Overlap lines
#    show a clean midpoint handoff instead of a single-edge sweep.
# f) Visual cleanliness:
#    We keep active lines separated, cap extras at one per line, and choose companion
#    positions before the opposite inner marker so clues are not buried under heavy
#    repainting.
# g) Color inference:
#    No ordering uses color ids.  A companion still changes the paint color only for
#    its own ray, and the winning overlap color is determined by geometry (fewer
#    travel steps, then right/bottom tie priority), not by any hidden color ranking.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill inward periodic progressions launched from both opposite edges of one axis.

    Input structure:
    - bg is the most common color, with sparse non-bg dots on a subset of rows or columns.
    - One opposite-edge pair dominates: either left+right or top+bottom has more boundary
      dots than the other pair, and that pair determines whether we process rows or columns.
    - Every non-empty active line has two marker colors: one color appears exactly twice
      from the left/top side, and another color appears exactly twice from the right/bottom
      side; each pair consists of an edge dot plus one inward dot, and their gap is the step.
    - An optional extra dot on the line is either a companion aligned to exactly one source
      beyond its inner marker, or a bystander aligned to neither source.

    Rule:
    - Compare opposite-edge pair counts; left+right means row-wise processing, top+bottom
      means column-wise processing.
    - On each non-empty line, recover the two sources from the colors that occur exactly
      twice and touch the corresponding boundary.
    - Each source paints its beat positions: without a companion it spans the whole line,
      while with a companion it only spans from its boundary marker through that companion
      and uses the companion color as paint.
    - If both sources claim the same cell, the source that reaches it in fewer step-hops
      from its edge wins; equal-time meetings are resolved in favor of the right/bottom ray.

    Key insight:
    The new ambiguity on individual edges is intentional: direction is no longer chosen by a
    single boundary winner, but by the denser opposite-edge pair.  Once the two per-line
    marker pairs are recovered, the task becomes a line-local competition between two periodic
    waves whose overlap is resolved by travel distance and a fixed tie priority.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    H, W = g.shape

    row_pair = sum(1 for r in range(H) if g[r, 0] != bg) + sum(1 for r in range(H) if g[r, W - 1] != bg)
    col_pair = sum(1 for c in range(W) if g[0, c] != bg) + sum(1 for c in range(W) if g[H - 1, c] != bg)
    pair_counts = {"rows": row_pair, "cols": col_pair}
    max_count = max(pair_counts.values())
    assert max_count > 0, "No non-bg dots on any boundary pair"
    winners = [axis for axis, count in pair_counts.items() if count == max_count]
    assert len(winners) == 1, f"Ambiguous shooting axis: {winners} (counts={pair_counts})"

    by_row = winners[0] == "rows"
    num_lines = H if by_row else W
    line_len = W if by_row else H

    if by_row:
        sides = (
            {"name": "left", "edge_pos": 0, "sign": 1, "tie_priority": 0},
            {"name": "right", "edge_pos": line_len - 1, "sign": -1, "tie_priority": 1},
        )
    else:
        sides = (
            {"name": "top", "edge_pos": 0, "sign": 1, "tie_priority": 0},
            {"name": "bottom", "edge_pos": line_len - 1, "sign": -1, "tie_priority": 1},
        )

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

        sources = []
        marker_colors: set[int] = set()
        for side in sides:
            candidates = []
            for color, positions in colors.items():
                ps = sorted(positions)
                if len(ps) != 2:
                    continue
                if side["edge_pos"] not in ps:
                    continue
                other = ps[0] if ps[1] == side["edge_pos"] else ps[1]
                if other == side["edge_pos"]:
                    continue
                candidates.append((color, ps))

            assert len(candidates) == 1, (
                f"Line {li} ({'row' if by_row else 'col'}) needs exactly one {side['name']} marker pair: "
                f"{candidates}, all colors={dict(colors)}"
            )
            marker_color, marker_positions = candidates[0]
            marker_colors.add(marker_color)
            sources.append(
                {
                    "marker_color": marker_color,
                    "positions": marker_positions,
                    **side,
                }
            )

        assert len(marker_colors) == 2, (
            f"Line {li} ({'row' if by_row else 'col'}) must have two distinct marker colors: "
            f"{sources}"
        )

        claims: dict[int, tuple[int, int, int]] = {}
        for source in sources:
            edge_pos = int(source["edge_pos"])
            sign = int(source["sign"])
            tie_priority = int(source["tie_priority"])
            marker_color = int(source["marker_color"])
            marker_positions = sorted(int(p) for p in source["positions"])
            inner = max(marker_positions) if sign > 0 else min(marker_positions)
            step = abs(inner - edge_pos)
            assert step > 0, f"Zero step on line {li} for source {source}"
            offset = edge_pos % step

            companions = []
            for color, positions in colors.items():
                if color in marker_colors:
                    continue
                for p in positions:
                    if p % step != offset:
                        continue
                    if sign * (p - inner) <= 0:
                        continue
                    companions.append((abs(p - inner), int(p), int(color)))

            assert len(companions) <= 1, (
                f"Line {li} ({'row' if by_row else 'col'}) has ambiguous companion for {source['name']}: "
                f"{companions}, all colors={dict(colors)}"
            )

            if companions:
                _, companion_pos, companion_color = companions[0]
                paint = companion_color
                lo, hi = sorted((edge_pos, companion_pos))
            else:
                paint = marker_color
                lo, hi = 0, line_len - 1

            for p in range(lo, hi + 1):
                if p % step != offset:
                    continue
                travel = abs(p - edge_pos) // step
                prev = claims.get(p)
                if prev is None or travel < prev[0] or (travel == prev[0] and tie_priority > prev[1]):
                    claims[p] = (travel, tie_priority, paint)

        for p, (_, _, paint) in claims.items():
            out[cell(li, p)] = paint

    return out


# --------------- generator ---------------

_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 8)
_STEP_RANGE = (1, 5)
_COLORS = list(range(1, 10))

_MAX_RETRIES = 400
_LINE_BUILD_RETRIES = 80


def _spaced_indices(n: int, total: int) -> list[int]:
    """Choose n indices from [0, total) with a gap of at least 1 between neighbors."""
    available = total - (n - 1)
    if available < n:
        return []
    picks = sorted(random.sample(range(available), n))
    return [picks[i] + i for i in range(n)]


def _collision_steps(line_len: int) -> list[int]:
    """Steps that create a real overlap zone, not just four marker positions."""
    return [
        step
        for step in range(max(2, _STEP_RANGE[0]), _STEP_RANGE[1] + 1)
        if (line_len - 1) % step == 0 and 3 * step < line_len - 1
    ]


def _choose_dims(axis: str, n_active: int) -> tuple[int, int, int, int] | None:
    """Pick dimensions that fit the active lines and allow at least one overlap step."""
    min_num_lines = 2 * n_active - 1
    if axis == "rows":
        h_lo = max(_ROWS_RANGE[0], min_num_lines)
        if h_lo > _ROWS_RANGE[1]:
            return None
        H = random.randint(h_lo, _ROWS_RANGE[1])
        widths = [w for w in range(_COLS_RANGE[0], _COLS_RANGE[1] + 1) if _collision_steps(w)]
        if not widths:
            return None
        W = random.choice(widths)
        return H, W, H, W

    w_lo = max(_COLS_RANGE[0], min_num_lines)
    if w_lo > _COLS_RANGE[1]:
        return None
    W = random.randint(w_lo, _COLS_RANGE[1])
    heights = [h for h in range(_ROWS_RANGE[0], _ROWS_RANGE[1] + 1) if _collision_steps(h)]
    if not heights:
        return None
    H = random.choice(heights)
    return H, W, W, H


def _all_beats(edge_pos: int, step: int, line_len: int, sign: int) -> list[int]:
    if sign > 0:
        return list(range(edge_pos, line_len, step))
    return list(range(edge_pos, -1, -step))


def _aligned(edge_pos: int, step: int, pos: int) -> bool:
    return pos % step == edge_pos % step


def _pick_steps(line_len: int, force_collision: bool) -> tuple[int, int] | None:
    if force_collision:
        steps = _collision_steps(line_len)
        if not steps:
            return None
        step = random.choice(steps)
        return step, step

    for _ in range(50):
        left = random.randint(*_STEP_RANGE)
        right = random.randint(*_STEP_RANGE)
        if left + right >= line_len - 1:
            continue
        return left, right
    return None


def _build_line_dots(
    line_len: int,
    palette: list[int],
    kind: str,
) -> list[tuple[int, int]] | None:
    """Return [(pos, color), ...] for one active line, or None if this attempt is invalid."""
    force_collision = kind == "collision"
    steps = _pick_steps(line_len, force_collision=force_collision)
    if steps is None:
        return None
    left_step, right_step = steps
    left_edge, left_inner = 0, left_step
    right_edge, right_inner = line_len - 1, line_len - 1 - right_step
    if left_inner >= right_inner:
        return None

    if len(palette) < 3:
        return None
    left_color, right_color = random.sample(palette, 2)
    used_colors = {left_color, right_color}
    occupied = {left_edge, left_inner, right_inner, right_edge}
    dots = [
        (left_edge, left_color),
        (left_inner, left_color),
        (right_inner, right_color),
        (right_edge, right_color),
    ]

    left_beats = _all_beats(left_edge, left_step, line_len, 1)
    right_beats = _all_beats(right_edge, right_step, line_len, -1)

    if kind == "left_companion":
        candidates = [
            p
            for p in left_beats[2:]
            if p < right_inner and p not in occupied and not _aligned(right_edge, right_step, p)
        ]
        if not candidates:
            return None
        extra_pos = random.choice(candidates)
        extra_color = random.choice([c for c in palette if c not in used_colors])
        dots.append((extra_pos, extra_color))
    elif kind == "right_companion":
        candidates = [
            p
            for p in right_beats[2:]
            if p > left_inner and p not in occupied and not _aligned(left_edge, left_step, p)
        ]
        if not candidates:
            return None
        extra_pos = random.choice(candidates)
        extra_color = random.choice([c for c in palette if c not in used_colors])
        dots.append((extra_pos, extra_color))
    elif kind == "bystander":
        candidates = [
            p
            for p in range(line_len)
            if p not in occupied
            and not _aligned(left_edge, left_step, p)
            and not _aligned(right_edge, right_step, p)
        ]
        if not candidates:
            return None
        extra_pos = random.choice(candidates)
        extra_color = random.choice([c for c in palette if c not in used_colors])
        dots.append((extra_pos, extra_color))
    elif kind not in {"collision", "marker_only"}:
        return None

    return dots


def _count_pair_dots(dots_rc: list[tuple[int, int]], H: int, W: int) -> dict[str, int]:
    return {
        "rows": sum(1 for r, c in dots_rc if c == 0 or c == W - 1),
        "cols": sum(1 for r, c in dots_rc if r == 0 or r == H - 1),
    }


def _old_logic_would_fail(inp: np.ndarray) -> bool:
    """The original single-edge solver should not see a unique winning edge here."""
    bg = int(get_bg(inp))
    H, W = inp.shape
    counts = {
        "top": sum(1 for c in range(W) if inp[0, c] != bg),
        "bottom": sum(1 for c in range(W) if inp[H - 1, c] != bg),
        "left": sum(1 for r in range(H) if inp[r, 0] != bg),
        "right": sum(1 for r in range(H) if inp[r, W - 1] != bg),
    }
    max_count = max(counts.values())
    return sum(1 for v in counts.values() if v == max_count) != 1


def _gen_attempt() -> np.ndarray | None:
    axis = random.choice(("rows", "cols"))
    n_active = random.randint(*_ACTIVE_LINES_RANGE)
    dims = _choose_dims(axis, n_active)
    if dims is None:
        return None
    H, W, num_lines, line_len = dims

    active_lines = _spaced_indices(n_active, num_lines)
    if not active_lines:
        return None

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if axis == "rows" else (pos, line)

    palette = random.sample(_COLORS, random.randint(5, 7))
    kinds = ["collision"]
    if n_active > 1:
        kinds.append(random.choice(["left_companion", "right_companion"]))
    while len(kinds) < n_active:
        kinds.append(
            random.choices(
                ["collision", "marker_only", "left_companion", "right_companion", "bystander"],
                weights=[0.30, 0.25, 0.15, 0.15, 0.15],
                k=1,
            )[0]
        )
    random.shuffle(kinds)

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []

    for li, kind in zip(active_lines, kinds):
        dots = None
        for _ in range(_LINE_BUILD_RETRIES):
            dots = _build_line_dots(line_len, palette, kind)
            if dots is not None:
                break
        if dots is None:
            return None

        for pos, color in dots:
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

    pair_counts = _count_pair_dots(all_rc, H, W)
    if axis == "rows":
        if pair_counts["rows"] <= pair_counts["cols"]:
            return None
    else:
        if pair_counts["cols"] <= pair_counts["rows"]:
            return None

    if not _old_logic_would_fail(inp):
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
