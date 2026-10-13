from __future__ import annotations

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


NON_BLACK_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)
GEN_ATTEMPTS = 400


def _switch_color(row: np.ndarray) -> int | None:
    color = int(row[0])
    if color == Color.BLACK:
        return None
    return color if np.all(row == color) else None


def _switch_rows_by_color(grid: np.ndarray) -> dict[int, list[int]]:
    switches: dict[int, list[int]] = {}
    for r in range(grid.shape[0]):
        color = _switch_color(grid[r])
        if color is not None:
            switches.setdefault(color, []).append(r)
    return switches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use full-width monochrome rows as per-color switches that blank only that color inside open intervals.

    Input structure:
    - The background is black.
    - Some rows are filled edge-to-edge with one non-black color; those full rows are switch rows.
    - Other rows contain scattered pixels or small objects in several colors, and different switch-row colors may be interleaved vertically.

    Rule:
    - Scan rows from top to bottom and keep an independent on/off state for each color, starting from off.
    - When a full-width row of color X is encountered, keep that row unchanged and flip only X's state for later rows.
    - On non-switch rows, any pixel of color X lying in an interval where X is on is recolored to black; all other pixels stay unchanged.

    Key insight:
    - A switch affects only its own color and only between successive occurrences along the scanning direction, so interleaved colors must be tracked independently.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    active: dict[int, bool] = {}
    for r in range(g.shape[0]):
        color = _switch_color(g[r])
        if color is not None:
            active[color] = not active.get(color, False)
            continue

        row = g[r]
        for value in np.unique(row):
            color = int(value)
            if color != Color.BLACK and active.get(color, False):
                out[r, row == color] = Color.BLACK
    return out


def _partition_positive(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    assert total >= parts >= 1
    values = np.ones(parts, dtype=int)
    for _ in range(total - parts):
        values[int(rng.integers(parts))] += 1
    return [int(v) for v in values]


def _sample_color_plan(rng: np.random.Generator) -> tuple[list[int], dict[int, int], int, int, int]:
    n_switch_colors = int(rng.integers(2, 4))
    all_colors = list(map(int, rng.choice(NON_BLACK_COLORS, size=n_switch_colors + 1, replace=False)))
    switch_colors = all_colors[:n_switch_colors]
    passive_color = all_colors[-1]
    even_color = switch_colors[0]
    odd_color = switch_colors[1]

    counts = {even_color: 2, odd_color: int(rng.choice([1, 3]))}
    for color in switch_colors[2:]:
        counts[color] = int(rng.choice([1, 2]))
    return all_colors, counts, even_color, odd_color, passive_color


def _is_interleaved(seq: list[int], color: int) -> bool:
    positions = [i for i, value in enumerate(seq) if value == color]
    return len(positions) >= 2 and positions[-1] - positions[0] >= 2


def _sample_switch_sequence(counts: dict[int, int], even_color: int, rng: np.random.Generator) -> list[int]:
    bag = [color for color, count in counts.items() for _ in range(count)]
    for _ in range(300):
        seq = bag.copy()
        rng.shuffle(seq)
        if any(seq[i] == seq[i + 1] for i in range(len(seq) - 1)):
            continue
        if not _is_interleaved(seq, even_color):
            continue
        return seq
    raise RuntimeError("failed to sample interleaved switch sequence")


def _sample_layout(total_switches: int, n_colors: int, rng: np.random.Generator) -> tuple[int, int, list[int], dict[int, tuple[int, int]]]:
    min_h = max(12, 2 * total_switches + 1)
    max_h = min(24, min_h + 6)
    h = int(rng.integers(min_h, max_h + 1))

    band_widths = [int(rng.integers(2, 4)) for _ in range(n_colors)]
    min_w = max(10, sum(band_widths) + n_colors + 1)
    max_w = min(20, min_w + 5)
    w = int(rng.integers(min_w, max_w + 1))

    row_gaps = _partition_positive(h - total_switches, total_switches + 1, rng)
    switch_rows = [row_gaps[0]]
    cursor = row_gaps[0]
    for i in range(1, total_switches):
        cursor += 1 + row_gaps[i]
        switch_rows.append(cursor)

    col_gaps = _partition_positive(w - sum(band_widths), n_colors + 1, rng)
    bands: dict[int, tuple[int, int]] = {}
    cursor = col_gaps[0]
    for idx in range(n_colors):
        bands[idx] = (cursor, cursor + band_widths[idx] - 1)
        cursor += band_widths[idx] + col_gaps[idx + 1]
    return h, w, switch_rows, bands


def _segments_for_color(height: int, color_switch_rows: list[int], all_switch_rows: set[int]) -> list[tuple[list[int], bool]]:
    segments: list[tuple[list[int], bool]] = []
    prev = -1
    for phase, stop in enumerate(color_switch_rows + [height]):
        rows = [r for r in range(prev + 1, stop) if r not in all_switch_rows]
        segments.append((rows, bool(phase % 2)))
        prev = stop
    return segments


def _paint_run(grid: np.ndarray, r: int, band: tuple[int, int], color: int, rng: np.random.Generator) -> None:
    c0, c1 = band
    width = c1 - c0 + 1
    length_cap = width if rng.random() < 0.3 else min(2, width)
    length = int(rng.integers(1, length_cap + 1))
    start = int(rng.integers(c0, c1 - length + 2))
    grid[r, start : start + length] = color


def _state_by_row(height: int, row_to_switch: dict[int, int], colors: list[int]) -> dict[int, np.ndarray]:
    states = {color: np.zeros(height, dtype=bool) for color in colors}
    active = {color: False for color in colors}
    for r in range(height):
        switch = row_to_switch.get(r)
        if switch is not None:
            active[switch] = not active[switch]
            continue
        for color in colors:
            states[color][r] = active[color]
    return states


def _wrong_delete_below_first(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    switches = _switch_rows_by_color(out)
    first_rows = {color: rows[0] for color, rows in switches.items()}
    switch_rows = {r for rows in switches.values() for r in rows}
    for r in range(out.shape[0]):
        if r in switch_rows:
            continue
        for color, first in first_rows.items():
            if r > first:
                out[r, out[r] == color] = Color.BLACK
    return out


def _wrong_global_open(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    active = False
    for r in range(g.shape[0]):
        switch = _switch_color(g[r])
        if switch is not None:
            active = not active
            continue
        if active:
            out[r, out[r] != Color.BLACK] = Color.BLACK
    return out


def _validate_example(grid: np.ndarray, passive_color: int, even_color: int) -> bool:
    out = main(grid)
    switches = _switch_rows_by_color(grid)
    switch_rows = {r for rows in switches.values() for r in rows}

    if np.array_equal(out, grid):
        return False
    if passive_color in switches:
        return False
    if even_color not in switches or len(switches[even_color]) != 2:
        return False
    if not any(len(rows) % 2 == 1 for rows in switches.values()):
        return False

    removed = (grid != out) & (out == Color.BLACK)
    if not np.any(removed):
        return False

    row_to_switch = {r: color for color, rows in switches.items() for r in rows}
    states = _state_by_row(grid.shape[0], row_to_switch, list({*switches.keys(), passive_color}))

    if np.any((grid == passive_color) & (out != passive_color)):
        return False

    even_removed = np.any((grid == even_color) & states[even_color][:, None] & (out == Color.BLACK))
    even_kept_after_second = any(
        np.any((grid[r] == even_color) & (out[r] == even_color))
        for r in range(switches[even_color][1] + 1, grid.shape[0])
        if r not in switch_rows
    )
    if not even_removed or not even_kept_after_second:
        return False

    mixed_row_found = False
    for r in range(grid.shape[0]):
        if r in switch_rows:
            continue
        present = sorted(set(map(int, grid[r])) - {Color.BLACK})
        if len(present) < 2:
            continue
        changed = {color for color in present if np.any((grid[r] == color) & (out[r] == Color.BLACK))}
        kept = {color for color in present if np.any((grid[r] == color) & (out[r] == color))}
        if changed and kept:
            mixed_row_found = True
            break
    if not mixed_row_found:
        return False

    if np.array_equal(out, _wrong_delete_below_first(grid)):
        return False
    if np.array_equal(out, _wrong_global_open(grid)):
        return False

    for r in range(grid.shape[0]):
        color = _switch_color(grid[r])
        if color is None and np.all(grid[r] == grid[r, 0]) and int(grid[r, 0]) != Color.BLACK:
            return False
        if color is not None and not np.array_equal(out[r], grid[r]):
            return False
    return True


def _build_candidate(rng: np.random.Generator) -> tuple[np.ndarray, int, int]:
    colors, counts, even_color, _odd_color, passive_color = _sample_color_plan(rng)
    switch_colors = [color for color in colors if color in counts]
    sequence = _sample_switch_sequence(counts, even_color, rng)

    h, w, switch_rows, band_slots = _sample_layout(len(sequence), len(colors), rng)
    bands = {color: band_slots[idx] for idx, color in enumerate(colors)}
    grid = np.zeros((h, w), dtype=int)

    row_to_switch = dict(zip(switch_rows, sequence))
    for r, color in row_to_switch.items():
        grid[r, :] = color

    switch_rows_by_color: dict[int, list[int]] = {}
    for r, color in row_to_switch.items():
        switch_rows_by_color.setdefault(color, []).append(r)
    all_switch_rows = set(switch_rows)

    for color in colors:
        color_switch_rows = switch_rows_by_color.get(color, [])
        segments = _segments_for_color(h, color_switch_rows, all_switch_rows)
        base_placements = 2 if color == passive_color else 1
        for rows, _active in segments:
            if not rows:
                continue
            count = base_placements + int(len(rows) >= 3 and rng.random() < 0.35)
            count = min(count, len(rows))
            chosen_rows = list(map(int, rng.choice(rows, size=count, replace=False)))
            for r in chosen_rows:
                _paint_run(grid, r, bands[color], color, rng)

    colors_with_switches = list(switch_rows_by_color)
    states = _state_by_row(h, row_to_switch, colors)
    mixed_pairs = [(a, passive_color) for a in colors_with_switches if np.any(states[a] != states[passive_color])]
    if mixed_pairs:
        active_color, kept_color = mixed_pairs[int(rng.integers(len(mixed_pairs)))]
        candidates = [
            r
            for r in range(h)
            if r not in all_switch_rows and states[active_color][r] and not states[kept_color][r]
        ]
        if candidates:
            demo_row = int(rng.choice(candidates))
            _paint_run(grid, demo_row, bands[active_color], active_color, rng)
            _paint_run(grid, demo_row, bands[kept_color], kept_color, rng)

    return grid, passive_color, even_color


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()
        grid, passive_color, even_color = _build_candidate(rng)
        if _validate_example(grid, passive_color, even_color):
            return grid
    raise RuntimeError("failed to generate a valid switch-row puzzle")
