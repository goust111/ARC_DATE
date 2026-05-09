# Combo variant of 16de56c4 (index 18)
# combo_spec: D10/2+D15/0
# Name: 全部为伴随+多伴随点

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(_ROOT / "solvers"))
from arc_utils import get_bg


# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior and trace:
# - Preserved scaffold from the base task/spec:
#   one boundary wins the non-bg count race, so it defines the shooting edge;
#   each active row/column on that axis contains exactly one repeated marker
#   color with two dots, and the gap from the edge dot to the inner dot is the
#   beat step.
# - D10/2 removes the bystander notion entirely: every foreign dot on that line
#   must affect the ray, even when it is off the beat class.
# - D15/0 requires 2 or 3 such foreign dots per active line, so one line now
#   contains several colored stopping points instead of 0/1 special point.
# - Example on a left-shooting row with marker M at t=0 and t=3:
#   companions A/B/C at t=5/6/10 produce beats t=0,3 -> A; t=6 -> B;
#   t=9 -> C.  In the original task only aligned companions mattered, so A
#   would be ignored and the result would be either all M or one single-color
#   segment.  In the fused task A still matters because order along the ray,
#   not modular alignment, decides the segments.
# b) Complexity calibration:
# - Companion count is capped at 2 or 3.  That is enough to create clearly
#   multi-colored rays without exploding the number of cases.
# - Each companion uses a distinct singleton color on its line, so the solver
#   never confuses a companion with a second marker.
# - The generator forces several off-beat companions per grid; otherwise the
#   fused task could visually collapse into the aligned-only multi-companion
#   variant and the D10 contribution would be invisible.
# c) Hidden assumption audit against the original solver:
# - The old "find the first aligned companion and break" assumption is invalid.
#   We must collect every non-marker dot, sort it by distance from the shooting
#   edge, and process segments in that order.
# - The old single [lo, hi] fill span is invalid because each consecutive
#   companion owns a different color interval.
# - The old distinction between aligned companion and off-beat bystander is
#   removed; every non-marker singleton becomes a segment endpoint.
# - The old write guard `cur == bg or cur == marker` remains valid because the
#   target companion cell should stay visible as an input dot, and later
#   segments start strictly after earlier endpoints in beat order.
# d) Placement legality enforced by the generator:
# - Each active line has exactly one repeated marker color, exactly 2 marker
#   dots, and exactly 2 or 3 singleton companions.
# - All companions are strictly beyond the inner marker in shooting direction,
#   never overlap existing dots, and avoid the opposite edge to keep boundary
#   counts unambiguous.
# - Companions are chosen from distinct beat buckets so every segment owns at
#   least one beat and is therefore visible in the output.
# e) Visual observability:
# - From 2-4 examples, a human can see that every foreign dot terminates a
#   colored section, even when that dot is not on the periodic beat.  The
#   ordering cue is geometric distance from the emitting edge, not color id.
# - Anti-degeneration checks compare the fused output with both the original
#   single-companion logic and an aligned-only multi-companion logic; generated
#   samples are rejected when the outputs match.
# f) Visual cleanliness:
# - Interaction stays on one sparse row/column at a time; active lines are
#   separated by blank gaps, and companions do not stack on the opposite edge.
# - Limiting to 2-3 companions prevents overly fragmented lines.
# g) New solver assumptions and generator support:
# - `main()` assumes every non-empty line has exactly one repeated color and
#   2-3 singleton companions; the generator constructs lines that way.
# - `main()` assumes companions are strictly ordered away from the edge after
#   distance normalization; the generator places unique positions only.
# - `main()` assumes each later segment still contains at least one beat; the
#   generator enforces distinct beat buckets for consecutive companions.


def _detect_edge(grid: np.ndarray, bg: int) -> str:
    H, W = grid.shape
    edge_counts = {
        "top": sum(1 for c in range(W) if grid[0, c] != bg),
        "bottom": sum(1 for c in range(W) if grid[H - 1, c] != bg),
        "left": sum(1 for r in range(H) if grid[r, 0] != bg),
        "right": sum(1 for r in range(H) if grid[r, W - 1] != bg),
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


def _segment_positions(prev_t: int, end_t: int, step: int, first: bool) -> range:
    start = 0 if first else ((prev_t // step) + 1) * step
    stop = (end_t // step) * step
    if stop < start:
        return range(0, 0)
    return range(start, stop + 1, step)


def _collect_line_colors(
    grid: np.ndarray,
    bg: int,
    by_row: bool,
    line_index: int,
    line_len: int,
) -> dict[int, list[int]]:
    def cell(pos: int) -> tuple[int, int]:
        return (line_index, pos) if by_row else (pos, line_index)

    colors: dict[int, list[int]] = defaultdict(list)
    for pos in range(line_len):
        value = int(grid[cell(pos)])
        if value != bg:
            colors[value].append(pos)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill periodic rays from one dominant edge, with every foreign dot acting as an ordered segment stop.

    Input structure:
    - The background is the most common color and the rest are sparse dots.
    - Exactly one boundary contains strictly more non-background dots than the
      other three; that boundary is the shooting edge, so rows are active for
      left/right shooting and columns are active for top/bottom shooting.
    - Each active shooting line contains exactly one repeated marker color with
      two dots: one on the shooting edge and one interior. Their distance is the
      step of the ray.
    - Every other dot on that same line is a companion. There are 2 or 3 of
      them, each with a unique color, and they may be on-beat or off-beat.

    Rule:
    - Detect the unique shooting edge, then normalize each line by distance from
      that edge.
    - On each non-empty line, the marker pair defines the beat positions
      t = 0, step, 2*step, ... .
    - Sort all non-marker dots by distance from the shooting edge. The first
      companion colors all beat cells from the edge up to its own depth; the
      next companion colors the later beat cells up to its depth; continue
      segment by segment. Nothing is painted beyond the last companion.
    - Companion dots themselves stay as placed input dots; only background cells
      and marker cells on those beats are recolored.

    Key insight:
    The repeated color no longer decides the final paint except for exposing the
    rhythm. The real control signal is the outward order of all foreign dots:
    every one of them terminates one segment, even when it is not aligned with
    the beat, so the output is recovered by walking the ray edge-outward and
    painting beat buckets with the endpoint companion colors.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    H, W = g.shape

    edge = _detect_edge(g, bg)
    by_row = edge in ("left", "right")
    forward, _ = _edge_info(edge, W if by_row else H)
    num_lines = H if by_row else W
    line_len = W if by_row else H

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    def to_t(pos: int) -> int:
        return pos if forward else (line_len - 1 - pos)

    def from_t(t: int) -> int:
        return t if forward else (line_len - 1 - t)

    for line_index in range(num_lines):
        colors = _collect_line_colors(g, bg, by_row, line_index, line_len)
        if not colors:
            continue

        repeating = {color: sorted(ps) for color, ps in colors.items() if len(ps) >= 2}
        assert len(repeating) == 1, (
            f"Line {line_index} ({'row' if by_row else 'col'}) must have exactly one "
            f"repeating marker color: {dict(colors)}"
        )

        marker, marker_positions = next(iter(repeating.items()))
        assert len(marker_positions) == 2, (
            f"Line {line_index} ({'row' if by_row else 'col'}) marker {marker} has "
            f"{len(marker_positions)} dots (expected 2): {marker_positions}"
        )

        marker_ts = sorted(to_t(pos) for pos in marker_positions)
        assert marker_ts[0] == 0, (
            f"Line {line_index} ({'row' if by_row else 'col'}) marker {marker} must "
            f"touch the shooting edge: {marker_positions}"
        )
        step = marker_ts[1]
        assert step > 0, f"Line {line_index} ({'row' if by_row else 'col'}) has zero step"

        companions: list[tuple[int, int]] = []
        for color, positions in colors.items():
            if color == marker:
                continue
            assert len(positions) == 1, (
                f"Line {line_index} ({'row' if by_row else 'col'}) non-marker color "
                f"{color} appears {len(positions)} times: {positions}"
            )
            t = to_t(positions[0])
            assert t > step, (
                f"Line {line_index} ({'row' if by_row else 'col'}) companion {color} "
                f"must lie beyond the inner marker: {positions[0]}"
            )
            companions.append((t, color))

        companions.sort(key=lambda item: item[0])
        assert 2 <= len(companions) <= 3, (
            f"Line {line_index} ({'row' if by_row else 'col'}) has {len(companions)} "
            f"companions (expected 2 or 3): {companions}"
        )

        prev_t = -1
        prev_bucket = 0
        for idx, (end_t, paint) in enumerate(companions):
            assert end_t > prev_t, (
                f"Line {line_index} ({'row' if by_row else 'col'}) companions are not "
                f"strictly ordered: {companions}"
            )
            assert (end_t // step) > prev_bucket, (
                f"Line {line_index} ({'row' if by_row else 'col'}) segment has no visible "
                f"beat between companions: {companions} with step={step}"
            )
            for t in _segment_positions(prev_t, end_t, step, idx == 0):
                rc = cell(line_index, from_t(t))
                cur = int(out[rc])
                if cur == bg or cur == marker:
                    out[rc] = paint
            prev_t = end_t
            prev_bucket = end_t // step

    return out


def _original_main_like(grid: np.ndarray) -> np.ndarray:
    """Original semantics: only the first aligned companion matters."""
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

    for line_index in range(num_lines):
        colors = _collect_line_colors(g, bg, by_row, line_index, line_len)
        if not colors:
            continue

        repeating = {color: sorted(ps) for color, ps in colors.items() if len(ps) >= 2}
        assert len(repeating) == 1
        marker, marker_positions = next(iter(repeating.items()))
        assert len(marker_positions) == 2
        a, b = sorted(marker_positions)
        step = b - a
        offset = a % step

        companion = None
        for color, positions in colors.items():
            if color == marker:
                continue
            if any(pos % step == offset for pos in positions):
                companion = color
                break

        if companion is None:
            paint = marker
            lo, hi = 0, line_len - 1
        else:
            paint = companion
            occupied = marker_positions + colors[companion]
            lo, hi = min(occupied), max(occupied)

        for pos in range(lo, hi + 1):
            if pos % step != offset:
                continue
            rc = cell(line_index, pos)
            cur = int(out[rc])
            if cur == bg or cur == marker:
                out[rc] = paint

    return out


def _aligned_only_multi_main_like(grid: np.ndarray) -> np.ndarray:
    """Reference semantics where only aligned companions segment the ray."""
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

    def ordered_positions(prev_anchor: int, target: int, step: int, first: bool) -> range:
        if forward:
            start = prev_anchor if first else prev_anchor + step
            return range(start, target + 1, step)
        start = prev_anchor if first else prev_anchor - step
        return range(start, target - 1, -step)

    for line_index in range(num_lines):
        colors = _collect_line_colors(g, bg, by_row, line_index, line_len)
        if not colors:
            continue

        repeating = {color: sorted(ps) for color, ps in colors.items() if len(ps) >= 2}
        assert len(repeating) == 1
        marker, marker_positions = next(iter(repeating.items()))
        assert len(marker_positions) == 2
        a, b = sorted(marker_positions)
        step = b - a
        offset = edge_pos % step

        aligned: list[tuple[int, int]] = []
        for color, positions in colors.items():
            if color == marker:
                continue
            pos = positions[0]
            if pos % step == offset:
                aligned.append((pos, color))

        aligned.sort(key=lambda item: item[0], reverse=not forward)
        prev_anchor = edge_pos
        for idx, (target_pos, paint) in enumerate(aligned):
            for pos in ordered_positions(prev_anchor, target_pos, step, idx == 0):
                rc = cell(line_index, pos)
                cur = int(out[rc])
                if cur == bg or cur == marker:
                    out[rc] = paint
            prev_anchor = target_pos

    return out


# --------------- generator ---------------

_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")
_PALETTE_SIZE_RANGE = (4, 7)

_MAX_RETRIES = 800


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


def _valid_bucket_choices(line_len: int, step: int) -> list[tuple[int, list[int]]]:
    choices: list[tuple[int, list[int]]] = []
    max_q = (line_len - 2) // step
    for q in range(1, max_q + 1):
        lo = max(step + 1, q * step)
        hi = min((q + 1) * step - 1, line_len - 2)
        if lo <= hi:
            choices.append((q, list(range(lo, hi + 1))))
    return choices


def _build_line_dots(
    edge: str,
    line_len: int,
    palette: list[int],
    marker_color: int,
    n_companions: int,
    require_offbeat: bool,
) -> tuple[list[tuple[int, int]], dict[str, int | bool]] | None:
    forward, edge_pos = _edge_info(edge, line_len)

    def from_t(t: int) -> int:
        return t if forward else (line_len - 1 - t)

    step_candidates = []
    for step in range(_STEP_RANGE[0], min(_STEP_RANGE[1], line_len - 2) + 1):
        buckets = _valid_bucket_choices(line_len, step)
        if len(buckets) < n_companions:
            continue
        if require_offbeat and not any(
            any(t % step != 0 for t in ts) for _, ts in buckets
        ):
            continue
        step_candidates.append((step, buckets))

    if not step_candidates:
        return None

    step, buckets = random.choice(step_candidates)
    dots = [(from_t(0), marker_color), (from_t(step), marker_color)]

    bucket_indices = list(range(len(buckets)))
    if require_offbeat:
        offbeat_indices = [
            idx
            for idx, (_, ts) in enumerate(buckets)
            if any(t % step != 0 for t in ts)
        ]
        forced_idx = random.choice(offbeat_indices)
        remaining = [idx for idx in bucket_indices if idx != forced_idx]
        chosen_indices = [forced_idx] + random.sample(remaining, n_companions - 1)
    else:
        chosen_indices = random.sample(bucket_indices, n_companions)

    chosen_indices.sort(key=lambda idx: buckets[idx][0])
    companion_colors = random.sample([c for c in palette if c != marker_color], n_companions)

    chosen_ts: list[int] = []
    has_offbeat = False
    forced_remaining = require_offbeat
    for color, bucket_idx in zip(companion_colors, chosen_indices):
        _, choices = buckets[bucket_idx]
        if forced_remaining:
            offbeat_choices = [t for t in choices if t % step != 0]
            t = random.choice(offbeat_choices)
            forced_remaining = False
        else:
            t = random.choice(choices)
        has_offbeat = has_offbeat or (t % step != 0)
        chosen_ts.append(t)
        dots.append((from_t(t), color))

    meta = {
        "step": step,
        "n_companions": n_companions,
        "has_offbeat": has_offbeat,
    }
    return dots, meta


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

    palette_size = random.randint(*_PALETTE_SIZE_RANGE)
    palette = random.sample(_COLORS, palette_size)

    min_offbeat = max(2, n_active // 2)
    offbeat_lines = set(random.sample(active_lines, random.randint(min_offbeat, n_active)))
    three_lines = set(random.sample(active_lines, random.randint(1, max(1, n_active // 2))))

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []
    shooting_edge_dots = 0
    offbeat_line_count = 0
    three_companion_line_count = 0

    for line_index in active_lines:
        marker_color = random.choice(palette)
        n_companions = 3 if line_index in three_lines else random.choice((2, 3))
        built = _build_line_dots(
            edge=edge,
            line_len=line_len,
            palette=palette,
            marker_color=marker_color,
            n_companions=n_companions,
            require_offbeat=line_index in offbeat_lines,
        )
        if built is None:
            return None
        dots, meta = built

        for pos, color in dots:
            r, c = cell(line_index, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

        shooting_edge_dots += 1
        offbeat_line_count += int(bool(meta["has_offbeat"]))
        three_companion_line_count += int(int(meta["n_companions"]) == 3)

    if offbeat_line_count < min_offbeat:
        return None
    if three_companion_line_count < 1:
        return None

    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(count >= shooting_edge_dots for count in other_counts.values()):
        return None

    try:
        out = main(inp)
        legacy_out = _original_main_like(inp)
        aligned_only_out = _aligned_only_multi_main_like(inp)
    except Exception:
        return None

    if np.array_equal(inp, out):
        return None
    if np.array_equal(out, legacy_out):
        return None
    if np.array_equal(out, aligned_only_out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
