from __future__ import annotations

from collections import Counter

import numpy as np


BACKGROUND = 0
BAR_COLOR = 1
MARKER_COLORS = (2, 3, 4)
GRAY = 5

RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _extract_bars(grid: np.ndarray) -> list[dict[str, int]]:
    h, w = grid.shape
    bars: list[dict[str, int]] = []
    for c in range(w):
        rows = np.flatnonzero(grid[:, c] == BAR_COLOR)
        if rows.size == 0:
            continue
        assert np.all(np.diff(rows) == 1), "each blue column must be contiguous"
        assert int(rows[-1]) == h - 1, "each blue bar must be bottom-aligned"
        height = int(rows.size)
        assert 3 <= height <= 10, "bar heights must stay in the task range"
        bars.append({"col": c, "top": int(rows[0]), "height": height})

    assert 5 <= len(bars) <= 7, "expected 5-7 blue bars"
    cols = [bar["col"] for bar in bars]
    assert cols == sorted(cols), "bars must be read left-to-right"
    for left, right in zip(cols, cols[1:]):
        gap = right - left - 1
        assert gap in (1, 2), "input gaps must be 1 or 2 columns"
    return bars


def _vote_counts(grid: np.ndarray, bar: dict[str, int]) -> dict[int, int]:
    counts = {color: 0 for color in MARKER_COLORS}
    c = bar["col"]
    for r in range(bar["top"], grid.shape[0]):
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) in MARKER_COLORS:
                counts[int(grid[nr, nc])] += 1
    return counts


def _ranked_bars(grid: np.ndarray) -> list[dict[str, int]]:
    bars = _extract_bars(grid)
    recolored: list[dict[str, int]] = []
    for bar in bars:
        counts = _vote_counts(grid, bar)
        assert sum(counts.values()) >= 1, "each bar must touch at least one marker"
        best = max(counts.values())
        winners = [color for color, count in counts.items() if count == best]
        assert len(winners) == 1, "adjacency vote must have a unique winner"
        recolored.append(
            {
                "col": bar["col"],
                "top": bar["top"],
                "height": bar["height"],
                "new_color": winners[0],
            }
        )

    totals = {
        color: sum(bar["height"] for bar in recolored if bar["new_color"] == color)
        for color in {bar["new_color"] for bar in recolored}
    }
    assert len(set(totals.values())) == len(totals), "recolored group totals must be unique"

    recolored.sort(key=lambda bar: (-totals[bar["new_color"]], -bar["height"], bar["col"]))
    return recolored


def _render_output(shape: tuple[int, int], bars: list[dict[str, int]]) -> np.ndarray:
    h, w = shape
    required_width = 2 * len(bars) - 1
    assert required_width <= w, "packed output bars must fit the grid width"

    out = np.full((h, w), BACKGROUND, dtype=int)
    out_col = 0
    for bar in bars:
        height = int(bar["height"])
        top = h - height
        split = top + (height + 1) // 2
        out[top:split, out_col] = int(bar["new_color"])
        out[split:h, out_col] = GRAY
        out_col += 2
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor blue bars by adjacent marker votes, sort them by recolored frequency, then gray their lower halves.

    Input structure:
    - The grid has a black background, 5-7 bottom-aligned one-cell-wide blue vertical bars, and isolated red/green/yellow marker pixels.
    - Bar columns are separated by 1-2 blank columns, while markers sit in the gaps or side blank space without touching each other by 4-neighborhood.

    Rule:
    - For each blue bar, count how many 4-neighbor side contacts it has with each marker color and recolor the whole bar with the uniquely most frequent marker color.
    - Sum bar heights within each recolored color group, sort groups by that total from large to small, and sort bars inside a group by height from tall to short.
    - Repack the bars from the left with a one-column gap between consecutive bars, keep them bottom-aligned, preserve the upper ceil(height/2) cells in the voted color, recolor the lower floor(height/2) cells gray, and remove all markers.

    Key insight:
    - The marker pixels do more than recolor local bars: their vote winners define color groups whose total pixel mass determines the final left-to-right order before the lower-half gray split is applied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"

    allowed = {BACKGROUND, BAR_COLOR, *MARKER_COLORS}
    assert set(map(int, np.unique(g))) <= allowed, "unexpected input colors"

    ranked = _ranked_bars(g)
    return _render_output(g.shape, ranked)


def _sample_layout(n_bars: int) -> tuple[int, list[int]]:
    extras = int(RNG.integers(0, n_bars + 2))
    slots = np.zeros(n_bars + 1, dtype=int)
    if extras:
        chosen = RNG.choice(np.arange(n_bars + 1), size=extras, replace=False)
        slots[chosen] = 1

    left_margin = int(slots[0])
    right_margin = int(slots[-1])
    gap_extras = slots[1:-1].tolist()

    cols = [left_margin]
    for gap_extra in gap_extras:
        cols.append(cols[-1] + 2 + int(gap_extra))
    width = cols[-1] + 1 + right_margin
    return width, cols


def _sample_assignments(heights: list[int]) -> list[int] | None:
    n_bars = len(heights)
    for _ in range(200):
        n_colors = 2 if RNG.random() < 0.45 else 3
        used_colors = RNG.choice(np.array(MARKER_COLORS), size=n_colors, replace=False).tolist()

        cut_points = sorted(RNG.choice(np.arange(1, n_bars), size=n_colors - 1, replace=False).tolist())
        bounds = [0] + cut_points + [n_bars]
        group_sizes = [bounds[i + 1] - bounds[i] for i in range(n_colors)]
        if max(group_sizes) < 2:
            continue

        assignment = []
        for color, size in zip(used_colors, group_sizes):
            assignment.extend([int(color)] * int(size))
        RNG.shuffle(assignment)

        totals = {
            color: sum(height for height, assigned in zip(heights, assignment) if assigned == color)
            for color in used_colors
        }
        if len(set(totals.values())) != len(totals):
            continue

        order = sorted(
            range(n_bars),
            key=lambda idx: (-totals[assignment[idx]], -heights[idx], idx),
        )
        if order == list(range(n_bars)):
            continue
        return assignment
    return None


def _exclusive_marker_candidates(grid: np.ndarray, bar: dict[str, int]) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    h = grid.shape[0]
    c = bar["col"]
    top = bar["top"]
    top_marker = (top - 1, c)
    assert top_marker[0] >= 0

    seen = {top_marker}
    side_candidates: list[tuple[int, int]] = []
    for r in range(top, h):
        for nc in (c - 1, c + 1):
            pos = (r, nc)
            if not _in_bounds(grid.shape, pos[0], pos[1]) or int(grid[pos]) != BACKGROUND:
                continue
            blue_neighbors = [
                (nr, nnc)
                for nr, nnc in _neighbors4(pos[0], pos[1])
                if _in_bounds(grid.shape, nr, nnc) and int(grid[nr, nnc]) == BAR_COLOR
            ]
            if blue_neighbors == [(r, c)] and pos not in seen:
                seen.add(pos)
                side_candidates.append(pos)
    return top_marker, side_candidates


def _can_place_marker(grid: np.ndarray, pos: tuple[int, int], allow_blue_touch: bool) -> bool:
    r, c = pos
    if not _in_bounds(grid.shape, r, c) or int(grid[r, c]) != BACKGROUND:
        return False
    for nr, nc in _neighbors4(r, c):
        if not _in_bounds(grid.shape, nr, nc):
            continue
        val = int(grid[nr, nc])
        if val in MARKER_COLORS:
            return False
        if not allow_blue_touch and val == BAR_COLOR:
            return False
    return True


def _place_marker(grid: np.ndarray, pos: tuple[int, int], color: int) -> None:
    assert _can_place_marker(grid, pos, allow_blue_touch=True)
    grid[pos] = int(color)


def _choose_extra_colors(winner: int, side_capacity: int) -> list[int]:
    others = [color for color in MARKER_COLORS if color != winner]
    profiles: list[list[int]] = [[]]
    if side_capacity >= 1:
        profiles.append([winner])
    if side_capacity >= 2:
        profiles.extend([[winner, winner], [winner, int(RNG.choice(np.array(others)))]])
    if side_capacity >= 3:
        other_pair = RNG.choice(np.array(others), size=2, replace=False).tolist()
        profiles.extend([[winner, winner, int(RNG.choice(np.array(others)))], [winner, other_pair[0], other_pair[1]]])
    if side_capacity >= 4:
        other_pair = RNG.choice(np.array(others), size=2, replace=False).tolist()
        profiles.append([winner, winner, other_pair[0], other_pair[1]])

    weights = np.array([1] * len(profiles), dtype=float)
    for idx, profile in enumerate(profiles):
        if any(color != winner for color in profile):
            weights[idx] = 2.6
        elif len(profile) >= 2:
            weights[idx] = 1.5
    weights /= weights.sum()
    return profiles[int(RNG.choice(np.arange(len(profiles)), p=weights))]


def _place_bar_extras(
    grid: np.ndarray,
    side_candidates: list[tuple[int, int]],
    extra_colors: list[int],
) -> bool:
    if not extra_colors:
        return True

    positions = side_candidates.copy()
    RNG.shuffle(positions)

    def backtrack(index: int, start: int) -> bool:
        if index == len(extra_colors):
            return True
        for pos_index in range(start, len(positions)):
            pos = positions[pos_index]
            if not _can_place_marker(grid, pos, allow_blue_touch=True):
                continue
            grid[pos] = int(extra_colors[index])
            if backtrack(index + 1, pos_index + 1):
                return True
            grid[pos] = BACKGROUND
        return False

    return backtrack(0, 0)


def _place_stray_markers(grid: np.ndarray, needed_by_color: dict[int, int]) -> bool:
    if not any(needed_by_color.values()):
        return True

    candidates = [
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if _can_place_marker(grid, (r, c), allow_blue_touch=False)
    ]
    RNG.shuffle(candidates)

    cursor = 0
    for color in MARKER_COLORS:
        for _ in range(needed_by_color[color]):
            placed = False
            while cursor < len(candidates):
                pos = candidates[cursor]
                cursor += 1
                if not _can_place_marker(grid, pos, allow_blue_touch=False):
                    continue
                grid[pos] = int(color)
                placed = True
                break
            if not placed:
                return False
    return True


def _independent_expected(grid: np.ndarray) -> np.ndarray:
    bars = _extract_bars(grid)
    recolored = []
    for bar in bars:
        counts = _vote_counts(grid, bar)
        winner = max(counts, key=lambda color: counts[color])
        assert counts[winner] > max(count for color, count in counts.items() if color != winner)
        recolored.append({"col": bar["col"], "height": bar["height"], "new_color": int(winner)})

    totals = {
        color: sum(bar["height"] for bar in recolored if bar["new_color"] == color)
        for color in {bar["new_color"] for bar in recolored}
    }
    assert len(set(totals.values())) == len(totals)

    recolored = sorted(recolored, key=lambda bar: (-totals[bar["new_color"]], -bar["height"], bar["col"]))
    return _render_output(grid.shape, recolored)


def _validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not set(map(int, np.unique(grid))) <= {BACKGROUND, BAR_COLOR, *MARKER_COLORS}:
        return False

    marker_counts = Counter(map(int, grid[np.isin(grid, MARKER_COLORS)].tolist()))
    if any(marker_counts[color] < 2 for color in MARKER_COLORS):
        return False

    for r, c in np.argwhere(np.isin(grid, MARKER_COLORS)):
        for nr, nc in _neighbors4(int(r), int(c)):
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) in MARKER_COLORS:
                return False

    bars = _extract_bars(grid)
    tallies: list[tuple[int, dict[int, int]]] = []
    has_counterexample_bar = False
    for bar in bars:
        counts = _vote_counts(grid, bar)
        if sum(counts.values()) < 1:
            return False
        winners = [color for color, count in counts.items() if count == max(counts.values())]
        if len(winners) != 1:
            return False
        if sum(count > 0 for count in counts.values()) >= 2:
            has_counterexample_bar = True
        tallies.append((winners[0], counts))
    if not has_counterexample_bar:
        return False

    assignments = [winner for winner, _counts in tallies]
    totals = {
        color: sum(bar["height"] for bar, assigned in zip(bars, assignments) if assigned == color)
        for color in set(assignments)
    }
    if len(set(totals.values())) != len(totals):
        return False

    sorted_order = sorted(
        range(len(bars)),
        key=lambda idx: (-totals[assignments[idx]], -bars[idx]["height"], idx),
    )
    if sorted_order == list(range(len(bars))):
        return False

    expected = _independent_expected(grid)
    actual = main(grid)
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    n_bars = int(RNG.integers(5, 8))
    height = int(RNG.integers(11, 16))
    width, cols = _sample_layout(n_bars)
    if width > 20:
        return None

    max_bar_height = min(10, height - 1)
    heights = RNG.integers(3, max_bar_height + 1, size=n_bars).tolist()
    if len(set(heights)) == 1:
        heights[int(RNG.integers(0, n_bars))] = int(RNG.integers(3, max_bar_height + 1))
    assignments = _sample_assignments(heights)
    if assignments is None:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    bars: list[dict[str, int]] = []
    for col, bar_height, winner in zip(cols, heights, assignments):
        top = height - int(bar_height)
        grid[top:height, int(col)] = BAR_COLOR
        bars.append({"col": int(col), "top": int(top), "height": int(bar_height), "winner": int(winner)})

    side_data: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    for bar in bars:
        top_marker, side_candidates = _exclusive_marker_candidates(grid, bar)
        if not _can_place_marker(grid, top_marker, allow_blue_touch=True):
            return None
        side_data.append((top_marker, side_candidates))

    for bar, (top_marker, _side_candidates) in zip(bars, side_data):
        _place_marker(grid, top_marker, bar["winner"])

    used_counterexample = False
    for bar, (_top_marker, side_candidates) in zip(bars, side_data):
        extra_colors = _choose_extra_colors(bar["winner"], len(side_candidates))
        if any(color != bar["winner"] for color in extra_colors):
            used_counterexample = True
        if not _place_bar_extras(grid, side_candidates, extra_colors):
            return None

    if not used_counterexample:
        return None

    marker_counts = Counter(map(int, grid[np.isin(grid, MARKER_COLORS)].tolist()))
    needed_by_color = {color: max(0, 2 - marker_counts[color]) for color in MARKER_COLORS}
    extra_noise = {color: 0 for color in MARKER_COLORS}
    for color in MARKER_COLORS:
        if RNG.random() < 0.35:
            extra_noise[color] += 1
    for color in MARKER_COLORS:
        needed_by_color[color] += extra_noise[color]

    if not _place_stray_markers(grid, needed_by_color):
        return None

    return grid if _validate_generated(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
