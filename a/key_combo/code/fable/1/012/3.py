from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
YELLOW = 4
BLOCK_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)

GEN_MAX_ATTEMPTS = 400
ROW_MAX_ATTEMPTS = 300


def get_bg(grid: np.ndarray) -> int:
    """Mirror the shared arc_utils background rule: use the most common color."""
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _row_segments(row: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    segments: list[tuple[int, int, int]] = []
    width = int(row.shape[0])
    c = 0
    while c < width:
        if int(row[c]) == bg:
            c += 1
            continue
        color = int(row[c])
        c0 = c
        while c < width and int(row[c]) == color:
            c += 1
        segments.append((color, c0, c - 1))
    return segments


def _simulate_positions(
    width: int,
    starts: list[int],
    widths: list[int],
    energy: int,
) -> tuple[list[int], dict[str, int | bool]]:
    starts = [int(v) for v in starts]
    widths = [int(v) for v in widths]
    energy = int(energy)

    collisions = 0
    boundary = False
    moved = False
    stalled = False
    active = 0

    while active < len(starts):
        block_w = widths[active]
        if energy < block_w:
            stalled = True
            break

        affordable = energy // block_w
        cur_end = starts[active] + block_w - 1

        if active == len(starts) - 1:
            distance = int(width - 1 - cur_end)
            step = min(affordable, distance)
            if step > 0:
                moved = True
            starts[active] += step
            energy -= step * block_w
            if step == distance:
                boundary = True
            else:
                stalled = True
            break

        next_start = starts[active + 1]
        gap = int(next_start - cur_end - 1)
        step = min(affordable, gap)
        if step > 0:
            moved = True
        starts[active] += step
        energy -= step * block_w
        if step == gap:
            collisions += 1
            active += 1
            continue
        stalled = True
        break

    summary: dict[str, int | bool] = {
        "collisions": collisions,
        "boundary": boundary,
        "moved": moved,
        "stalled": stalled,
        "remaining_energy": energy,
    }
    return starts, summary


def main(grid: np.ndarray) -> np.ndarray:
    """Use each leading yellow bar as an energy budget that pushes the row's blocks through a rightward collision chain.

    Input structure:
    - The grid has a dominant background and several separated track rows.
    - On each track row, the first foreground segment is a yellow energy bar.
    - To its right sit one or more solid horizontal blocks of width one or two,
      each in its own non-yellow color, with at least one background cell between
      neighboring blocks.

    Rule:
    - Solve each track row independently and keep the yellow bar fixed.
    - The bar length is that row's initial energy. Energy first drives the
      leftmost block; moving one cell costs the block's width.
    - A block stops as soon as it becomes adjacent to the next block, then
      transfers all leftover energy to that next block. If the current block
      cannot afford one more step, the row stops. If the active block reaches
      the right boundary, it stops there and any leftover energy disappears.

    Key insight:
    - This is a one-dimensional chain reaction, not a whole-row shift: only the
      currently active block consumes energy, and wider blocks spend energy
      faster, so final positions are determined by width-weighted costs and the
      visible gaps between blocks.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    bg = get_bg(g)
    out = g.copy()

    for r in range(height):
        segments = _row_segments(g[r], bg)
        if not segments or segments[0][0] != YELLOW:
            continue

        bar_color, bar_start, bar_end = segments[0]
        del bar_color
        block_segments = segments[1:]
        if not block_segments:
            continue

        starts: list[int] = []
        block_widths: list[int] = []
        block_colors: list[int] = []
        for color, c0, c1 in block_segments:
            block_w = int(c1 - c0 + 1)
            assert color != YELLOW, "only the leading bar may be yellow"
            assert block_w in (1, 2), "blocks must be width one or two"
            starts.append(int(c0))
            block_widths.append(block_w)
            block_colors.append(int(color))

        energy = int(bar_end - bar_start + 1)
        final_starts, _summary = _simulate_positions(width, starts, block_widths, energy)

        new_row = np.full(width, bg, dtype=int)
        new_row[bar_start : bar_end + 1] = YELLOW
        for start, block_w, color in zip(final_starts, block_widths, block_colors):
            new_row[start : start + block_w] = color
        out[r] = new_row

    return out


def _make_row_from_layout(
    width: int,
    energy: int,
    block_widths: list[int],
    inter_gaps: list[int],
    tail: int,
) -> tuple[np.ndarray, dict[str, int | bool]] | None:
    core = int(energy + sum(block_widths) + sum(inter_gaps) + tail)
    if core >= width:
        return None

    pre_gap = int(width - core)
    if pre_gap < 1:
        return None

    colors = random.sample(BLOCK_COLORS, len(block_widths))
    starts = [int(energy + pre_gap)]
    for idx in range(1, len(block_widths)):
        prev_start = starts[-1]
        prev_width = block_widths[idx - 1]
        starts.append(int(prev_start + prev_width + inter_gaps[idx - 1]))

    row = np.full(width, BLACK, dtype=int)
    row[:energy] = YELLOW
    for start, block_w, color in zip(starts, block_widths, colors):
        row[start : start + block_w] = color

    _final_starts, summary = _simulate_positions(width, starts, block_widths, energy)
    return row, summary


def _generate_chain_row(width: int, force_width: int | None) -> np.ndarray | None:
    for _ in range(ROW_MAX_ATTEMPTS):
        block_count = random.randint(3, 4)
        block_widths = [random.choice((1, 2)) for _ in range(block_count)]
        if force_width is not None and force_width not in block_widths:
            block_widths[random.randrange(block_count)] = int(force_width)

        min_energy = max(4, block_widths[0] + block_widths[1])
        if min_energy > 8:
            continue
        energy = random.randint(min_energy, 8)

        max_gap = (energy - block_widths[1]) // block_widths[0]
        if max_gap < 1:
            continue
        inter_gaps = [random.randint(1, max_gap)]
        remaining = energy - inter_gaps[0] * block_widths[0]

        stopped = False
        for idx in range(1, block_count - 1):
            cur_w = block_widths[idx]
            if not stopped and remaining >= cur_w and random.random() < 0.55:
                reachable = remaining // cur_w
                inter_gaps.append(random.randint(1, reachable))
                remaining -= inter_gaps[-1] * cur_w
            else:
                inter_gaps.append(max(1, remaining // cur_w + random.randint(1, 3)))
                stopped = True

        tail = random.randint(1, 4)
        built = _make_row_from_layout(width, energy, block_widths, inter_gaps, tail)
        if built is None:
            continue
        row, summary = built
        if bool(summary["moved"]) and int(summary["collisions"]) >= 1:
            return row
    return None


def _generate_stall_row(width: int, force_width: int | None) -> np.ndarray | None:
    for _ in range(ROW_MAX_ATTEMPTS):
        block_count = random.randint(2, 4)
        block_widths = [random.choice((1, 2)) for _ in range(block_count)]
        if force_width is not None and force_width not in block_widths:
            block_widths[random.randrange(block_count)] = int(force_width)

        min_energy = max(2, block_widths[0])
        energy = random.randint(min_energy, 7)
        first_gap = energy // block_widths[0] + random.randint(1, 3)
        inter_gaps = [first_gap] + [random.randint(1, 4) for _ in range(block_count - 2)]
        tail = random.randint(1, 4)

        built = _make_row_from_layout(width, energy, block_widths, inter_gaps, tail)
        if built is None:
            continue
        row, summary = built
        if (
            bool(summary["moved"])
            and bool(summary["stalled"])
            and not bool(summary["boundary"])
            and int(summary["collisions"]) == 0
        ):
            return row
    return None


def _generate_boundary_row(width: int, force_width: int | None) -> np.ndarray | None:
    for _ in range(ROW_MAX_ATTEMPTS):
        block_widths = [random.choice((1, 2)), random.choice((1, 2))]
        if force_width is not None and force_width not in block_widths:
            block_widths[random.randrange(2)] = int(force_width)

        min_energy = max(4, block_widths[0] + block_widths[1])
        if min_energy > 8:
            continue
        energy = random.randint(min_energy, 8)

        max_gap = (energy - block_widths[1]) // block_widths[0]
        if max_gap < 1:
            continue
        inter_gaps = [random.randint(1, max_gap)]
        remaining = energy - inter_gaps[0] * block_widths[0]
        max_tail = remaining // block_widths[1]
        if max_tail < 1:
            continue
        tail = random.randint(1, max_tail)

        built = _make_row_from_layout(width, energy, block_widths, inter_gaps, tail)
        if built is None:
            continue
        row, summary = built
        if bool(summary["moved"]) and bool(summary["boundary"]):
            return row
    return None


def _generate_row(width: int, scenario: str, force_width: int | None = None) -> np.ndarray | None:
    if scenario == "chain":
        return _generate_chain_row(width, force_width)
    if scenario == "stall":
        return _generate_stall_row(width, force_width)
    if scenario == "boundary":
        return _generate_boundary_row(width, force_width)
    raise ValueError(f"unknown scenario: {scenario}")


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        width = random.randint(17, 26)
        track_count = random.randint(3, 5)
        height = random.randint(2 * track_count + 1, min(2 * track_count + 5, 16))

        candidate_rows = list(range(0, height, 2))
        if len(candidate_rows) < track_count:
            continue
        track_rows = sorted(random.sample(candidate_rows, track_count))

        scenarios = ["chain", "stall", "boundary"]
        while len(scenarios) < track_count:
            scenarios.append(random.choice(("chain", "stall", "boundary")))
        random.shuffle(scenarios)

        force_widths = [None] * track_count
        force_widths[scenarios.index("chain")] = 2
        if "stall" in scenarios:
            force_widths[scenarios.index("stall")] = 1

        grid = np.full((height, width), BLACK, dtype=int)
        success = True
        for row_index, scenario, force_width in zip(track_rows, scenarios, force_widths):
            row = _generate_row(width, scenario, force_width)
            if row is None:
                success = False
                break
            grid[row_index] = row

        if not success:
            continue

        out = main(grid)
        if np.array_equal(out, grid):
            continue
        return grid

    raise RuntimeError("generate_input failed to build a valid energy-chain puzzle")
