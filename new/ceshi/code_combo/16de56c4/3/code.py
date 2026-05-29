# Combo variant of 16de56c4 (index 3)
# combo_spec: D3/2+D15/0
# Name: 颜色循环+多伴随点

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(_ROOT / "solvers"))
from arc_utils import get_bg


# Design analysis for the combo variant
# a) Concrete behavior and trace:
# - The base task had one repeated marker color per active line and at most one
#   aligned companion, so the whole fill on that line was either unbounded in the
#   marker color or one bounded monochrome segment in the companion color.
# - D15/0 changes the aligned-dot count: each active line now has 2 or 3 aligned
#   companions of different colors, ordered from the shooting edge outward.
#   These companions still split the beat scaffold into consecutive intervals.
# - D3/2 changes the fill mode: instead of using one constant paint color, the
#   ordered visible clue colors are consumed as a cycle. In this combo, the cycle
#   is taken from the aligned companions only, not from markers or bystanders.
# - Concrete row example with left shooting, step 2:
#   marker at 0/2 in color 4; companions at 4/8/12 in colors 7/3/6.
#   Old task: 0,2,4 -> 7; 6,8 -> 3; 10,12 -> 6 was impossible because only one
#   companion existed. D15-only: those three segments are uniformly 7, then 3,
#   then 6. Combo rule: companion order gives cycle [7,3,6]. Segment 1 paints
#   beats 0,2,4 as 7,3,(keep 7 at the companion); segment 2 paints 6,8 as 3,(keep
#   3); segment 3 paints 10,12 as 6,(keep 6). The result is still segmented, but
#   each segment shows the cycle rather than a flat color block.
# b) Complexity calibration:
# - Companion count is capped at 2 or 3, so the cycle has only 2 or 3 entries.
#   That is richer than the source task but still small enough to infer from a
#   few examples.
# - Every active line is forced to be multi-companion. That avoids collapsing
#   into the original task or into a dataset where the combo behavior appears on
#   only one exceptional line.
# - Optional bystanders remain sparse and off-beat, so they provide contrast
#   without competing with the core aligned structure.
# c) Implicit-assumption audit of the original solver:
# - The old "first aligned non-marker dot wins" assumption is invalid because the
#   line may contain several aligned companions; we must collect all of them in
#   trajectory order.
# - The old `paint` scalar per line is invalid because the combo needs a local
#   cycle, and the cycle phase depends on which interval is being filled.
# - The old single-span logic `lo, hi = min/max(...)` is insufficient because the
#   companions define several successive intervals, not one global bounded span.
# - The old write guard `cur == bg or cur == marker` still works and is important:
#   companions stay visible as the clues that define both the interval endpoints
#   and the cycle entries, while marker cells are allowed to recolor.
# d) Placement legality:
# - Each active line still has exactly one repeated marker color with exactly two
#   dots, one on the shooting edge and one one step inward.
# - Every aligned companion is a singleton, lies on the same beat class, and is
#   strictly beyond the inner marker along the shooting direction.
# - Companion colors are distinct within a line, preventing accidental repeated
#   non-marker colors that would confuse marker detection.
# - Optional bystanders are placed off-beat and off occupied cells, so they never
#   interfere with companion ordering or segment construction.
# e) Visual observability:
# - The shooting edge is still deducible from boundary-dot counts, and the beat
#   step is still revealed by the two repeated marker dots.
# - Humans can now read two coupled clues directly from the input: aligned
#   companion positions reveal the segment boundaries, while their left-to-right
#   or top-to-bottom order from the shooting edge reveals the cycle order.
# - No hidden ordering by color id is used; all ordering is geometric and visible.
# f) Visual cleanliness:
# - Output changes remain 1D sparse dotted rays on separated lines, so even with
#   more colors the scene stays legible.
# - Bystanders are rare and stay unchanged, preventing dense overlap in the same
#   region as the filled scaffold.
# g) Change inferability and anti-degeneration:
# - The generator rejects any sample whose output matches either the original
#   single-companion solver or a multi-companion-but-uniform-segment solver.
# - It also measures whether every active line contains at least one visibly
#   multi-colored segment on repaintable cells. This ensures the color-cycle part
#   of the combo is not merely present in metadata but actually visible.
# h) New solver assumptions and generator guarantees:
# - `main()` assumes every non-empty line has exactly one repeated marker color.
#   The generator guarantees this by giving every non-marker color only one dot.
# - `main()` assumes every active line has 2 or 3 aligned companions in visible
#   trajectory order beyond the inner marker. The generator builds them directly
#   from beat indices >= 2 and sorts only by spatial order.
# - `main()` assumes companion colors within a line are distinct. The generator
#   samples them without replacement.


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


def _edge_info(edge: str, line_len: int) -> tuple[bool, int]:
    forward = edge in ("left", "top")
    edge_pos = 0 if forward else line_len - 1
    return forward, edge_pos


def _segment_positions(prev_anchor: int, target: int, step: int, forward: bool, first: bool) -> list[int]:
    if forward:
        start = prev_anchor if first else prev_anchor + step
        return list(range(start, target + 1, step))
    start = prev_anchor if first else prev_anchor - step
    return list(range(start, target - 1, -step))


def _parse_line(
    g: np.ndarray,
    bg: int,
    li: int,
    by_row: bool,
    line_len: int,
    forward: bool,
    edge_pos: int,
) -> dict[str, object] | None:
    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    colors: dict[int, list[int]] = defaultdict(list)
    for p in range(line_len):
        v = int(g[cell(li, p)])
        if v != bg:
            colors[v].append(p)

    if not colors:
        return None

    repeating = [(color, positions) for color, positions in colors.items() if len(positions) >= 2]
    assert len(repeating) == 1, (
        f"Line {li} ({'row' if by_row else 'col'}) must have exactly one repeating color: {dict(colors)}"
    )
    marker, mpos = repeating[0]
    assert len(mpos) == 2, (
        f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
        f"has {len(mpos)} dots (expected 2): {mpos}"
    )

    ordered_marker = sorted(mpos, reverse=not forward)
    assert ordered_marker[0] == edge_pos, (
        f"Line {li} ({'row' if by_row else 'col'}) marker {marker} must touch the shooting edge: {mpos}"
    )
    inner_pos = ordered_marker[1]
    step = abs(inner_pos - edge_pos)
    assert step > 0, f"Line {li} has coincident marker dots: {mpos}"
    offset = edge_pos % step

    aligned_companions: list[tuple[int, int]] = []
    bystanders: list[tuple[int, int]] = []
    for color, positions in colors.items():
        if color == marker:
            continue
        assert len(positions) == 1, (
            f"Line {li} ({'row' if by_row else 'col'}) non-marker color {color} "
            f"appears multiple times: {positions}"
        )
        pos = positions[0]
        if pos % step == offset:
            if forward:
                assert pos > inner_pos, (
                    f"Line {li} aligned companion {color} at {pos} must lie beyond inner marker {inner_pos}"
                )
            else:
                assert pos < inner_pos, (
                    f"Line {li} aligned companion {color} at {pos} must lie beyond inner marker {inner_pos}"
                )
            aligned_companions.append((pos, color))
        else:
            bystanders.append((pos, color))

    aligned_companions.sort(key=lambda item: item[0], reverse=not forward)
    assert 2 <= len(aligned_companions) <= 3, (
        f"Line {li} ({'row' if by_row else 'col'}) has {len(aligned_companions)} aligned companions "
        f"(expected 2 or 3): {aligned_companions}"
    )
    cycle = [color for _, color in aligned_companions]
    assert len(set(cycle)) == len(cycle), (
        f"Line {li} ({'row' if by_row else 'col'}) companion colors must be distinct: {aligned_companions}"
    )

    return {
        "marker": marker,
        "step": step,
        "aligned_companions": aligned_companions,
        "bystanders": bystanders,
        "cycle": cycle,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill periodic rays from a unique shooting edge, using multi-companion intervals and a companion-color cycle.

    Input structure:
    - The background is the most frequent color and the non-background content is
      sparse. Exactly one boundary has strictly more non-background dots than the
      other three, which identifies the shooting edge.
    - Left/right shooting edges mean the active objects live on rows; top/bottom
      shooting edges mean they live on columns.
    - Every active shooting-direction line contains exactly one repeated marker
      color with exactly two dots: one on the shooting edge and one one step
      inward. Their gap defines the beat period.
    - Each active line also has 2 or 3 aligned companion singletons of different
      colors on the same beat class, ordered farther from the edge, plus optional
      off-beat bystanders that are not transformed.

    Rule:
    - Detect the unique shooting edge, then process each active row or column
      independently.
    - Recover the step from the repeated marker pair. Sort the aligned companion
      dots from the shooting edge outward; they split the beat scaffold into
      consecutive intervals ending at each companion.
    - Read the companion colors in that same outward order to form a cycle.
      Inside interval `j`, paint the beat cells from that interval with the cycle
      starting at companion `j`'s color and wrapping as needed.
    - Keep companion dots and bystanders unchanged; only background cells and the
      repeated marker cells are overwritten.

    Key insight:
    - The two repeated marker dots still reveal where the periodic ray runs, but
      the aligned companions now carry two jobs at once: their positions cut the
      ray into intervals, and their visible left-to-right or top-to-bottom order
      supplies the color cycle that must be replayed inside those intervals.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    H, W = g.shape

    edge = _detect_edge(g, bg)
    by_row = edge in ("left", "right")
    num_lines = H if by_row else W
    line_len = W if by_row else H
    forward, edge_pos = _edge_info(edge, line_len)

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    for li in range(num_lines):
        parsed = _parse_line(g, bg, li, by_row, line_len, forward, edge_pos)
        if parsed is None:
            continue

        marker = int(parsed["marker"])
        aligned_companions = list(parsed["aligned_companions"])
        cycle = list(parsed["cycle"])

        prev_anchor = edge_pos
        step = int(parsed["step"])
        for seg_idx, (target_pos, _) in enumerate(aligned_companions):
            beats = _segment_positions(prev_anchor, target_pos, step, forward, seg_idx == 0)
            for local_idx, p in enumerate(beats):
                paint = int(cycle[(seg_idx + local_idx) % len(cycle)])
                rc = cell(li, p)
                cur = int(out[rc])
                if cur == bg or cur == marker:
                    out[rc] = paint
            prev_anchor = target_pos

    return out


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    """Original single-companion uniform-fill rule, kept for anti-degeneration checks."""
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


def _segment_uniform_main(grid: np.ndarray) -> np.ndarray:
    """Multi-companion segmented rule without cycling, used to verify the D3 part of the combo."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    H, W = g.shape

    edge = _detect_edge(g, bg)
    by_row = edge in ("left", "right")
    num_lines = H if by_row else W
    line_len = W if by_row else H
    forward, edge_pos = _edge_info(edge, line_len)

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    for li in range(num_lines):
        parsed = _parse_line(g, bg, li, by_row, line_len, forward, edge_pos)
        if parsed is None:
            continue

        marker = int(parsed["marker"])
        step = int(parsed["step"])
        aligned_companions = list(parsed["aligned_companions"])
        prev_anchor = edge_pos
        for seg_idx, (target_pos, color) in enumerate(aligned_companions):
            beats = _segment_positions(prev_anchor, target_pos, step, forward, seg_idx == 0)
            for p in beats:
                rc = cell(li, p)
                cur = int(out[rc])
                if cur == bg or cur == marker:
                    out[rc] = int(color)
            prev_anchor = target_pos

    return out


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    """Summarize whether the generated input visibly exercises both combo dimensions."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    edge = _detect_edge(g, bg)
    by_row = edge in ("left", "right")
    num_lines = H if by_row else W
    line_len = W if by_row else H
    forward, edge_pos = _edge_info(edge, line_len)

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    active_lines = 0
    multi_lines = 0
    multicolor_segments = 0

    for li in range(num_lines):
        parsed = _parse_line(g, bg, li, by_row, line_len, forward, edge_pos)
        if parsed is None:
            continue
        active_lines += 1

        aligned_companions = list(parsed["aligned_companions"])
        cycle = list(parsed["cycle"])
        marker = int(parsed["marker"])
        step = int(parsed["step"])

        if len(aligned_companions) >= 2:
            multi_lines += 1

        prev_anchor = edge_pos
        for seg_idx, (target_pos, _) in enumerate(aligned_companions):
            beats = _segment_positions(prev_anchor, target_pos, step, forward, seg_idx == 0)
            visible_colors = []
            for local_idx, p in enumerate(beats):
                rc = cell(li, p)
                cur = int(g[rc])
                if cur == bg or cur == marker:
                    visible_colors.append(int(cycle[(seg_idx + local_idx) % len(cycle)]))
            if len(set(visible_colors)) >= 2:
                multicolor_segments += 1
            prev_anchor = target_pos

    fused = main(g)
    segmented = _segment_uniform_main(g)
    legacy = _legacy_main(g)
    return {
        "active_lines": active_lines,
        "multi_lines": multi_lines,
        "multicolor_segments": multicolor_segments,
        "cycle_delta": int(np.count_nonzero(fused != segmented)),
        "legacy_delta": int(np.count_nonzero(fused != legacy)),
    }


# --------------- generator ---------------

_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_MULTI_COMPANION_COUNTS = (2, 3)
_PALETTE_SIZE_RANGE = (5, 7)
_P_BYSTANDER = 0.35

_MAX_RETRIES = 500


def _compatible_active_range(edge: str) -> tuple[int, int] | None:
    by_row = edge in ("left", "right")
    max_num_lines = _ROWS_RANGE[1] if by_row else _COLS_RANGE[1]
    lo = _ACTIVE_LINES_RANGE[0]
    hi = min(_ACTIVE_LINES_RANGE[1], (max_num_lines + 1) // 2)
    if lo > hi:
        return None
    return lo, hi


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


def _beat_positions(edge: str, line_len: int, step: int) -> list[int]:
    forward, edge_pos = _edge_info(edge, line_len)
    if forward:
        return list(range(edge_pos, line_len, step))
    return list(range(edge_pos, -1, -step))


def _pick_step(line_len: int, n_companions: int) -> int | None:
    candidates = []
    for step in range(_STEP_RANGE[0], min(_STEP_RANGE[1], line_len - 1) + 1):
        beat_count = 1 + (line_len - 1) // step
        if beat_count >= n_companions + 2:
            candidates.append(step)
    if not candidates:
        return None
    return random.choice(candidates)


def _available_bystander_positions(edge: str, line_len: int, step: int, occupied: set[int]) -> list[int]:
    forward, edge_pos = _edge_info(edge, line_len)
    beat_set = set(_beat_positions(edge, line_len, step))
    far_edge = line_len - 1 if forward else 0
    candidates = []
    for p in range(line_len):
        if p in occupied or p in beat_set:
            continue
        if p == far_edge:
            continue
        candidates.append(p)
    return candidates


def _choose_companion_positions(edge: str, line_len: int, step: int, n_companions: int) -> list[int] | None:
    beats = _beat_positions(edge, line_len, step)
    if len(beats) < n_companions + 2:
        return None

    candidate_indices = list(range(2, len(beats)))
    if len(candidate_indices) > n_companions:
        candidate_indices = candidate_indices[:-1]
    if len(candidate_indices) < n_companions:
        candidate_indices = list(range(2, len(beats)))
    if len(candidate_indices) < n_companions:
        return None

    chosen = sorted(random.sample(candidate_indices, n_companions))
    return [beats[idx] for idx in chosen]


def _build_line_dots(
    edge: str,
    line_len: int,
    marker_color: int,
    palette: list[int],
) -> list[tuple[int, int]] | None:
    n_companions = random.choice(_MULTI_COMPANION_COUNTS)
    step = _pick_step(line_len, n_companions)
    if step is None:
        return None

    beats = _beat_positions(edge, line_len, step)
    edge_pos = beats[0]
    inner_pos = beats[1]
    companion_positions = _choose_companion_positions(edge, line_len, step, n_companions)
    if companion_positions is None:
        return None

    available_colors = [color for color in palette if color != marker_color]
    if len(available_colors) < n_companions:
        return None
    companion_colors = random.sample(available_colors, n_companions)

    dots = [(edge_pos, marker_color), (inner_pos, marker_color)]
    occupied = {edge_pos, inner_pos}
    for pos, color in zip(companion_positions, companion_colors):
        if pos in occupied:
            return None
        dots.append((pos, color))
        occupied.add(pos)

    if random.random() < _P_BYSTANDER:
        bystander_positions = _available_bystander_positions(edge, line_len, step, occupied)
        bystander_colors = [color for color in palette if color not in {c for _, c in dots}]
        if bystander_positions and bystander_colors:
            dots.append((random.choice(bystander_positions), random.choice(bystander_colors)))

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
    edge = random.choice(_EDGES)
    active_range = _compatible_active_range(edge)
    if active_range is None:
        return None
    n_active = random.randint(*active_range)
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

    palette_size = random.randint(*_PALETTE_SIZE_RANGE)
    palette = random.sample(_COLORS, palette_size)

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []
    shooting_edge_dots = 0

    for li in active_lines:
        marker_color = random.choice(palette)
        dots = _build_line_dots(edge, line_len, marker_color, palette)
        if dots is None:
            return None

        seen_positions: set[int] = set()
        for pos, color in dots:
            if pos in seen_positions:
                return None
            seen_positions.add(pos)
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
        segmented = _segment_uniform_main(inp)
        metrics = _fusion_metrics(inp)
    except Exception:
        return None

    if np.array_equal(inp, out):
        return None
    if np.array_equal(out, legacy):
        return None
    if np.array_equal(out, segmented):
        return None
    if metrics["active_lines"] != n_active:
        return None
    if metrics["multi_lines"] != n_active:
        return None
    if metrics["multicolor_segments"] < n_active:
        return None
    if metrics["cycle_delta"] < n_active:
        return None
    if metrics["legacy_delta"] < n_active:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
