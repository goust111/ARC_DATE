from __future__ import annotations

from collections import Counter

import numpy as np


_RNG = np.random.default_rng()
_PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
_GEN_RETRIES = 400


def _full_span_bars(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    rows: list[tuple[int, int]] = []
    cols: list[tuple[int, int]] = []
    h, w = grid.shape

    for r in range(h):
        color = int(grid[r, 0])
        if bool(np.all(grid[r, :] == color)):
            rows.append((r, color))
    for c in range(w):
        color = int(grid[0, c])
        if bool(np.all(grid[:, c] == color)):
            cols.append((c, color))
    return rows, cols


def _majority_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    order = np.argsort(-counts)
    if len(order) >= 2 and int(counts[order[0]]) == int(counts[order[1]]):
        raise ValueError("line majority is not unique")
    return int(colors[order[0]])


def _compress_runs(seq: list[int]) -> list[int]:
    out: list[int] = []
    for color in seq:
        if not out or out[-1] != color:
            out.append(int(color))
    return out


def _extract_sequence(
    grid: np.ndarray, bar_orientation: str, ignored_indices: list[int] | tuple[int, ...]
) -> list[int]:
    g = np.asarray(grid, dtype=int)

    if bar_orientation == "vertical":
        keep = [c for c in range(g.shape[1]) if c not in set(map(int, ignored_indices))]
        if not keep:
            raise ValueError("no non-bar columns left")
        line_colors = [_majority_color(g[r, keep]) for r in range(g.shape[0])]
        return _compress_runs(line_colors)

    if bar_orientation == "horizontal":
        keep = [r for r in range(g.shape[0]) if r not in set(map(int, ignored_indices))]
        if not keep:
            raise ValueError("no non-bar rows left")
        line_colors = [_majority_color(g[keep, c]) for c in range(g.shape[1])]
        return _compress_runs(line_colors)

    raise ValueError("invalid bar orientation")


def _segment_counts(sequence: list[int]) -> dict[int, int]:
    return dict(Counter(map(int, sequence)))


def _histogram_from_counts(counts: dict[int, int]) -> np.ndarray:
    if not counts:
        return np.zeros((0, 0), dtype=int)

    heights = list(counts.values())
    if len(set(heights)) != len(heights):
        raise ValueError("segment counts must be unique")

    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    out = np.zeros((max(heights), len(ordered)), dtype=int)
    for c, (color, height) in enumerate(ordered):
        out[: int(height), c] = int(color)
    return out


def _solve_counts(grid: np.ndarray) -> tuple[str, list[int], list[int], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    row_bars, col_bars = _full_span_bars(g)

    if bool(row_bars) == bool(col_bars):
        raise ValueError("expected full-span bars in exactly one orientation")

    if row_bars:
        ignored = [r for r, _color in row_bars]
        sequence = _extract_sequence(g, "horizontal", ignored)
        counts = _segment_counts(sequence)
        return "horizontal", ignored, sequence, counts

    ignored = [c for c, _color in col_bars]
    sequence = _extract_sequence(g, "vertical", ignored)
    counts = _segment_counts(sequence)
    return "vertical", ignored, sequence, counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect the full-span guide bar, compress the orthogonal color bands, then turn segment counts into a sorted top-down histogram.

    Input structure:
    - The grid is fully covered by colored bands rather than a single shared background.
    - Exactly one orientation contains 1 or 2 full-span monochrome bars: either whole rows or whole columns.
    - Aside from those bars, the scene consists of three band colors whose boundaries are slightly jagged.

    Rule:
    - Find every monochrome row or column that spans the whole grid and infer the guide-bar orientation.
    - If the bars are columns, ignore those columns and take the majority color of each remaining row; if the bars are rows, ignore those rows and take the majority color of each remaining column.
    - Compress consecutive equal majority colors into one sequence, then count how many separate segments each color occupies in that compressed sequence.
    - Sort colors by segment count from largest to smallest and output a histogram whose width is the number of colors and whose height is the largest segment count; each column is filled from the top downward with its color for exactly that many cells, and the rest stays black.

    Key insight:
    - The bars are not part of the final histogram; they only tell you which axis to compress along, and the histogram height measures fragmentation into separated runs, not area.
    """
    _orientation, _ignored, _sequence, counts = _solve_counts(np.asarray(grid, dtype=int))
    return _histogram_from_counts(counts)


def _random_run_sequence(count_map: dict[int, int]) -> list[int]:
    colors = list(count_map)
    total = sum(int(v) for v in count_map.values())
    remaining = {int(color): int(count) for color, count in count_map.items()}
    seq: list[int] = []

    def backtrack(last: int | None) -> bool:
        if len(seq) == total:
            return True

        choices = [color for color in colors if remaining[color] > 0 and color != last]
        _RNG.shuffle(choices)
        for color in choices:
            remaining[color] -= 1
            pending = list(remaining.values())
            max_pending = max(pending, default=0)
            if max_pending <= sum(pending) - max_pending + 1:
                seq.append(color)
                if backtrack(color):
                    return True
                seq.pop()
            remaining[color] += 1
        return False

    if not backtrack(None):
        raise RuntimeError("failed to build non-adjacent run sequence")
    return seq


def _random_positive_sum(parts: int, total: int) -> list[int]:
    values = np.ones(parts, dtype=int)
    for _ in range(int(total) - parts):
        values[int(_RNG.integers(0, parts))] += 1
    return values.tolist()


def _line_specs(
    sequence: list[int], heights: list[int], bar_color: int, usable_width: int, bar_width: int
) -> list[tuple[int, int, str, int]]:
    max_intrude = usable_width - (usable_width // 2 + 1)
    assert max_intrude >= 1

    critical_runs = [i for i, color in enumerate(sequence) if color != bar_color]
    critical_run = int(_RNG.choice(critical_runs)) if critical_runs and bar_width == 2 else None
    specs: list[tuple[int, int, str, int]] = []

    for run_idx, base_color in enumerate(sequence):
        run_len = int(heights[run_idx])
        last_side = "left" if bool(_RNG.integers(0, 2)) else "right"
        candidates = []
        if run_idx > 0:
            candidates.append(int(sequence[run_idx - 1]))
        if run_idx + 1 < len(sequence):
            candidates.append(int(sequence[run_idx + 1]))
        candidates = [c for c in candidates if c != base_color]
        fallback = [c for c in sequence if c != base_color]
        for _ in range(run_len):
            if run_idx == critical_run:
                intrude_color = int(bar_color)
                intrude_len = int(max_intrude)
            else:
                if candidates and _RNG.random() < 0.75:
                    intrude_color = int(candidates[int(_RNG.integers(0, len(candidates)))])
                else:
                    intrude_color = int(fallback[int(_RNG.integers(0, len(fallback)))])
                if intrude_color == base_color:
                    intrude_color = int(bar_color) if base_color != bar_color else int(fallback[0])
                intrude_len = int(_RNG.integers(1, max_intrude + 1))

            if _RNG.random() < 0.35:
                last_side = "left" if last_side == "right" else "right"
            specs.append((int(base_color), int(intrude_color), last_side, int(intrude_len)))
    return specs


def _paint_vertical_case(
    row_specs: list[tuple[int, int, str, int]],
    usable_width: int,
    bar_width: int,
    bar_color: int,
    bar_side: str,
) -> np.ndarray:
    h = len(row_specs)
    w = usable_width + bar_width
    grid = np.zeros((h, w), dtype=int)

    if bar_side == "left":
        band_slice = slice(bar_width, w)
        bar_slice = slice(0, bar_width)
    else:
        band_slice = slice(0, usable_width)
        bar_slice = slice(usable_width, w)

    for r, (base_color, intrude_color, side, intrude_len) in enumerate(row_specs):
        grid[r, band_slice] = int(base_color)
        band = grid[r, band_slice]
        if side == "left":
            band[:intrude_len] = int(intrude_color)
        else:
            band[-intrude_len:] = int(intrude_color)
        grid[r, bar_slice] = int(bar_color)

    return grid


def _sequence_output(sequence: list[int]) -> np.ndarray:
    return _histogram_from_counts(_segment_counts(sequence))


def _area_order_differs(grid: np.ndarray, colors: list[int], seg_counts: dict[int, int]) -> bool:
    color_areas = {color: int(np.sum(grid == color)) for color in colors}
    pairs = [(a, b) for a in colors for b in colors if a != b]
    for a, b in pairs:
        if seg_counts[a] > seg_counts[b] and color_areas[a] < color_areas[b]:
            return True
    return False


def _validate_generated(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30

    orientation, ignored, sequence, counts = _solve_counts(grid)
    row_bars, col_bars = _full_span_bars(grid)
    bars = row_bars if row_bars else col_bars

    assert 1 <= len(bars) <= 2
    assert len(counts) == 3
    assert sorted(counts.values()) == [1, 2, 3]

    bar_color = int(bars[0][1])
    assert all(int(color) == bar_color for _idx, color in bars)
    assert bar_color in counts

    out = main(grid)
    assert out.shape == (3, 3)

    other_orientation = "horizontal" if orientation == "vertical" else "vertical"
    wrong_sequence = _extract_sequence(grid, other_orientation, [])
    wrong_out = _sequence_output(wrong_sequence)
    assert not np.array_equal(out, wrong_out)

    if len(ignored) == 2:
        try:
            with_bars = _extract_sequence(grid, orientation, [])
        except ValueError:
            with_bars = []
        assert with_bars != sequence

    seq_colors = list(counts)
    assert _area_order_differs(grid, seq_colors, counts)

    if orientation == "vertical":
        assert not row_bars
    else:
        assert not col_bars


def _gen_vertical_case() -> np.ndarray | None:
    colors = list(_RNG.choice(_PALETTE, size=3, replace=False))
    seg_map = {int(color): count for color, count in zip(colors, _RNG.permutation([1, 2, 3]))}
    sequence = _random_run_sequence(seg_map)

    bar_width = 2 if _RNG.random() < 0.8 else 1
    usable_width = int(_RNG.choice([7, 9, 11] if bar_width == 2 else [8, 10, 12]))
    total_height = int(_RNG.integers(10, 16))
    heights = _random_positive_sum(len(sequence), total_height)

    bar_color = int(colors[int(_RNG.integers(0, len(colors)))])
    specs = _line_specs(sequence, heights, bar_color, usable_width, bar_width)
    bar_side = "left" if _RNG.random() < 0.5 else "right"
    grid = _paint_vertical_case(specs, usable_width, bar_width, bar_color, bar_side)

    if _RNG.random() < 0.5:
        grid = grid.T.copy()

    try:
        _validate_generated(grid)
    except (AssertionError, ValueError):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_vertical_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
