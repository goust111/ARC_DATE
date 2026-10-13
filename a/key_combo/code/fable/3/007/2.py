from __future__ import annotations

from dataclasses import dataclass
import itertools
import random

import numpy as np


BACKGROUND = 0
BARRIER_COLOR = 5
BAR_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 500


@dataclass(frozen=True)
class Barrier:
    row: int
    gap: tuple[int, int]
    color: int


@dataclass(frozen=True)
class Bar:
    row: int
    span: tuple[int, int]
    color: int


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, val in enumerate(mask.tolist()):
        if val and start is None:
            start = i
        elif not val and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _detect_barriers(g: np.ndarray, bg: int) -> list[Barrier]:
    h, w = g.shape
    barriers: list[Barrier] = []
    for r in range(h):
        row = g[r]
        if int(row[0]) == bg or int(row[-1]) == bg:
            continue
        gap_runs = _true_runs(row == bg)
        if len(gap_runs) != 1:
            continue
        g0, g1 = gap_runs[0]
        if g0 == 0 or g1 == w - 1:
            continue
        solid = row[row != bg]
        if solid.size == 0 or len(set(map(int, solid))) != 1:
            continue
        if solid.size < 4:
            continue
        barriers.append(Barrier(r, (int(g0), int(g1)), int(solid[0])))

    assert barriers, "expected at least one full-width barrier row with one interior gap"
    return sorted(barriers, key=lambda b: b.row)


def _detect_bars(g: np.ndarray, bg: int, barrier_rows: set[int]) -> list[Bar]:
    bars: list[Bar] = []
    colors: set[int] = set()
    rows: set[int] = set()
    for r in range(g.shape[0]):
        if r in barrier_rows:
            continue
        row = g[r]
        if not np.any(row != bg):
            continue
        runs = _true_runs(row != bg)
        assert len(runs) == 1, "each source row must contain exactly one horizontal bar"
        c0, c1 = runs[0]
        segment = row[c0 : c1 + 1]
        unique = set(map(int, segment))
        assert len(unique) == 1, "each horizontal bar must be solid and monochrome"
        color = next(iter(unique))
        assert c1 - c0 + 1 >= 2, "horizontal bars must have visible length"
        assert color not in colors, "bar colors must be distinct"
        assert r not in rows, "bar rows must be distinct"
        colors.add(color)
        rows.add(r)
        bars.append(Bar(int(r), (int(c0), int(c1)), int(color)))

    assert bars, "expected at least one falling horizontal bar"
    return bars


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _contained(span: tuple[int, int], gap: tuple[int, int]) -> bool:
    return gap[0] <= span[0] and span[1] <= gap[1]


def _simulate(
    shape: tuple[int, int],
    barriers: list[Barrier],
    bars: list[Bar],
    *,
    settled_blocks: bool = True,
) -> tuple[list[Bar], list[dict[str, object]]]:
    h, _w = shape
    settled: list[Bar] = []
    events: list[dict[str, object]] = []

    for bar in sorted(bars, key=lambda x: x.row, reverse=True):
        blockers: list[tuple[int, str, object]] = []
        for barrier in barriers:
            if barrier.row <= bar.row:
                continue
            if not _contained(bar.span, barrier.gap):
                blockers.append((barrier.row, "barrier", barrier))
        if settled_blocks:
            for other in settled:
                if other.row > bar.row and _overlap(bar.span, other.span):
                    blockers.append((other.row, "settled", other))

        if blockers:
            block_row, reason, blocker = min(blockers, key=lambda x: x[0])
            final_row = int(block_row) - 1
        else:
            reason = "bottom"
            blocker = None
            final_row = h - 1

        assert final_row >= bar.row, "a falling bar cannot move upward"
        final = Bar(final_row, bar.span, bar.color)
        if settled_blocks:
            for old in settled:
                assert final.row != old.row or not _overlap(final.span, old.span), "settled bars may not overlap"
        settled.append(final)
        events.append(
            {
                "color": bar.color,
                "start_row": bar.row,
                "final_row": final_row,
                "span": bar.span,
                "reason": reason,
                "blocker": blocker,
            }
        )

    return settled, events


def main(grid: np.ndarray) -> np.ndarray:
    """Drop horizontal bars through barrier gaps, with settled bars becoming new blockers.

    Input structure:
    - The background contains several one-cell-high colored horizontal bars in distinct upper rows.
    - Below them are full-width one-cell barrier lines, each with one interior background gap.
    - Barrier gaps have different column intervals, and bar spans can be contained in, overlap, or miss those gaps.

    Rule:
    - Process bars in descending initial row order, from lower bars to higher bars.
    - A bar falls vertically without changing its column span; it passes a barrier only when that barrier's gap interval fully contains the bar interval.
    - If the nearest lower blocker is a failed barrier or an already settled overlapping bar, the falling bar stops one row above it; otherwise it lands on the bottom row.
    - Original bar positions are erased, settled bars are painted at their final rows, and barrier lines remain fixed.

    Key insight:
    - The test is interval containment, not width comparison, and every settled bar narrows the channel for later bars in the low-to-high chain reaction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background(g)
    barriers = _detect_barriers(g, bg)
    bars = _detect_bars(g, bg, {b.row for b in barriers})

    out = g.copy()
    for bar in bars:
        c0, c1 = bar.span
        out[bar.row, c0 : c1 + 1] = bg

    settled, _events = _simulate(g.shape, barriers, bars, settled_blocks=True)
    for bar in settled:
        c0, c1 = bar.span
        assert np.all(out[bar.row, c0 : c1 + 1] == bg), "final bar position must be empty"
        out[bar.row, c0 : c1 + 1] = bar.color

    return out


def _choose_barrier_rows(height: int, count: int, bar_count: int) -> list[int] | None:
    first_min = bar_count + 5
    choices = [
        rows
        for rows in itertools.combinations(range(first_min, height - 1), count)
        if all(rows[i + 1] - rows[i] >= 2 for i in range(count - 1))
    ]
    if not choices:
        return None
    return list(random.choice(choices))


def _choose_gaps(width: int, count: int) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    channel_width = random.randint(8, min(10, width - 6))
    ch0 = random.randint(2, width - channel_width - 3)
    channel = (ch0, ch0 + channel_width - 1)
    used_widths: set[int] = set()
    used_starts: set[int] = set()
    gaps: list[tuple[int, int]] = []

    for _ in range(count):
        candidates: list[tuple[int, int]] = []
        for gap_width in range(channel_width, min(width - 4, channel_width + 6) + 1):
            if gap_width in used_widths:
                continue
            left_min = max(1, channel[1] - gap_width + 1)
            left_max = min(channel[0], width - gap_width - 1)
            for g0 in range(left_min, left_max + 1):
                if g0 in used_starts:
                    continue
                candidates.append((g0, g0 + gap_width - 1))
        if not candidates:
            return None
        gap = random.choice(candidates)
        gaps.append(gap)
        used_widths.add(gap[1] - gap[0] + 1)
        used_starts.add(gap[0])

    return channel, gaps


def _disjoint_from(span: tuple[int, int], spans: list[tuple[int, int]], gap: int = 0) -> bool:
    s0, s1 = span
    for a0, a1 in spans:
        if not (s1 + gap < a0 or a1 + gap < s0):
            return False
    return True


def _required_spans(
    width: int,
    channel: tuple[int, int],
    top_gap: tuple[int, int],
) -> list[tuple[int, int]] | None:
    ch0, ch1 = channel
    seed_width = random.choice([2, 3])
    seed0 = random.randint(ch0, ch0 + 1)
    seed = (seed0, seed0 + seed_width - 1)

    chain_candidates = []
    for length in (2, 3, 4):
        for c0 in range(ch0, ch1 - length + 2):
            span = (c0, c0 + length - 1)
            if span != seed and _overlap(span, seed):
                chain_candidates.append(span)
    if not chain_candidates:
        return None
    chain = random.choice(chain_candidates)

    clear_candidates = []
    for length in (2, 3):
        for c0 in range(ch0, ch1 - length + 2):
            span = (c0, c0 + length - 1)
            if _disjoint_from(span, [seed, chain], gap=1):
                clear_candidates.append(span)
    if not clear_candidates:
        return None
    clear = random.choice(clear_candidates)

    gap_width = top_gap[1] - top_gap[0] + 1
    if top_gap[1] + 1 < width:
        shifted = (top_gap[0] + 1, top_gap[1] + 1)
    elif top_gap[0] > 0:
        shifted = (top_gap[0] - 1, top_gap[1] - 1)
    else:
        return None
    if _contained(shifted, top_gap):
        return None

    return [seed, chain, clear, shifted]


def _extra_span(width: int, channel: tuple[int, int], top_gap: tuple[int, int]) -> tuple[int, int] | None:
    kind = random.choice(["settled", "barrier"])
    if kind == "settled":
        length = random.choice([2, 3])
        c0 = random.randint(channel[0], channel[1] - length + 1)
        return c0, c0 + length - 1

    for _ in range(40):
        length = random.randint(2, min(5, width - 4))
        c0 = random.randint(1, width - length - 2)
        span = (c0, c0 + length - 1)
        if not _contained(span, top_gap):
            return span
    return None


def _render_input(
    height: int,
    width: int,
    barrier_rows: list[int],
    gaps: list[tuple[int, int]],
    bars: list[Bar],
) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    for row, gap in zip(barrier_rows, gaps):
        grid[row, :] = BARRIER_COLOR
        grid[row, gap[0] : gap[1] + 1] = BACKGROUND
    for bar in bars:
        c0, c1 = bar.span
        assert np.all(grid[bar.row, c0 : c1 + 1] == BACKGROUND)
        grid[bar.row, c0 : c1 + 1] = bar.color
    return grid


def _semantic_checks(inp: np.ndarray) -> bool:
    try:
        bg = _background(inp)
        barriers = _detect_barriers(inp, bg)
        bars = _detect_bars(inp, bg, {b.row for b in barriers})
        out = main(inp)
        settled, events = _simulate(inp.shape, barriers, bars, settled_blocks=True)
        independent, _ = _simulate(inp.shape, barriers, bars, settled_blocks=False)
    except AssertionError:
        return False

    if inp.shape[0] > 30 or inp.shape[1] > 30 or out.shape[0] > 30 or out.shape[1] > 30:
        return False
    if not np.any(inp != out):
        return False
    if len(bars) < 4 or len(barriers) < 2:
        return False
    if len({b.gap[1] - b.gap[0] + 1 for b in barriers}) != len(barriers):
        return False
    if len({b.gap[0] for b in barriers}) != len(barriers):
        return False

    reasons = {str(e["reason"]) for e in events}
    if not {"bottom", "barrier", "settled"}.issubset(reasons):
        return False

    by_color_independent = {bar.color: bar.row for bar in independent}
    chain_events = [
        e
        for e in events
        if e["reason"] == "settled" and int(e["final_row"]) != by_color_independent[int(e["color"])]
    ]
    if not chain_events:
        return False

    equal_width_miss = False
    for event in events:
        if event["reason"] != "barrier":
            continue
        blocker = event["blocker"]
        if not isinstance(blocker, Barrier):
            continue
        span = event["span"]
        assert isinstance(span, tuple)
        if span[1] - span[0] == blocker.gap[1] - blocker.gap[0] and not _contained(span, blocker.gap):
            equal_width_miss = True
            break
    if not equal_width_miss:
        return False

    final_rows = {bar.row for bar in settled}
    if len(final_rows) < 3:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bar_count = random.randint(4, 6)
    height = random.randint(20, 30)
    width = random.randint(16, 28)
    barrier_count = random.randint(2, 4)

    barrier_rows = _choose_barrier_rows(height, barrier_count, bar_count)
    if barrier_rows is None:
        return None
    gap_choice = _choose_gaps(width, barrier_count)
    if gap_choice is None:
        return None
    channel, gaps = gap_choice

    required = _required_spans(width, channel, gaps[0])
    if required is None:
        return None

    available_rows = list(range(1, barrier_rows[0] - 1))
    if len(available_rows) < bar_count:
        return None
    chosen_rows = sorted(random.sample(available_rows, bar_count), reverse=True)
    colors = random.sample(BAR_COLORS, bar_count)

    spans = list(required)
    while len(spans) < bar_count:
        span = _extra_span(width, channel, gaps[0])
        if span is None:
            return None
        spans.append(span)

    bars = [Bar(row, span, color) for row, span, color in zip(chosen_rows, spans, colors)]
    inp = _render_input(height, width, barrier_rows, gaps, bars)
    return inp if _semantic_checks(inp) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid falling-span filter input")
