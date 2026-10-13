from __future__ import annotations

import random
import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 400


def _parse_scene(grid: np.ndarray) -> tuple[int, set[int], list[tuple[int, int]], dict[int, dict[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 3 and w >= 3, "grid too small"

    bg = int(g[1, 0])
    assert np.all(g[1, :] == bg), "row 1 must be the empty isolation band"

    top = g[0]
    legend_cols = np.flatnonzero(top != bg)
    assert legend_cols.size > 0, "missing top-row legend"
    c0 = int(legend_cols[0])
    c1 = int(legend_cols[-1])
    assert legend_cols.size == c1 - c0 + 1, "legend must be one contiguous bar"
    assert np.all(top[:c0] == bg) and np.all(top[c1 + 1 :] == bg), "legend must be isolated on the top row"
    legend_colors = set(map(int, top[c0 : c1 + 1]))
    assert bg not in legend_colors, "legend colors must be non-background"

    seeds: list[tuple[int, int]] = []
    obstacles: dict[int, dict[int, int]] = {}
    for r in range(2, h):
        row_obstacles: dict[int, int] = {}
        for c in range(w):
            value = int(g[r, c])
            if value == bg:
                continue
            if c == 0:
                seeds.append((r, value))
            else:
                row_obstacles[c] = value
        if row_obstacles:
            obstacles[r] = row_obstacles

    assert seeds, "expected at least one left-edge seed"
    seed_rows = {r for r, _ in seeds}
    assert len(seed_rows) == len(seeds), "seed rows must be unique"

    return bg, legend_colors, sorted(seeds), obstacles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each left-edge beam rightward, treating legend colors as walls and all other obstacle colors as dyes.

    Input structure:
    - The top row contains one contiguous color-bar legend, and the whole second row is background as an isolation band.
    - Below the isolation band, the leftmost column holds one-pixel beam seeds; other non-background cells are one-pixel obstacles.
    - Obstacles may use colors that also appear in the legend, so role is determined by legend membership rather than appearance alone.

    Rule:
    - Each seed emits a horizontal beam to the right, painting only background cells with the beam's current color.
    - When the beam reaches an obstacle whose color is listed in the legend, it stops immediately before that obstacle and leaves the obstacle unchanged.
    - When the beam reaches an obstacle whose color is not listed in the legend, the obstacle stays in place, the beam passes through it, and from the cell to its right onward the beam continues in the obstacle's color.
    - A beam can be recolored multiple times in sequence; the legend row, isolation row, seeds, and obstacles all remain unchanged.

    Key insight:
    - The crucial distinction is not the obstacle's visible color by itself but whether that color belongs to the top legend: legend-member colors are walls, while non-member colors act as recoloring dyes.
    """
    g = np.asarray(grid, dtype=int)
    bg, legend_colors, seeds, _obstacles = _parse_scene(g)
    _h, w = g.shape
    out = g.copy()

    for r, current_color in seeds:
        beam_color = int(current_color)
        for c in range(1, w):
            value = int(g[r, c])
            if value == bg:
                out[r, c] = beam_color
            else:
                if value in legend_colors:
                    break
                beam_color = value

    return out


def _sample_positions(width: int, count: int, *, last_needs_tail: bool) -> list[int] | None:
    min_gap = 2
    min_first = 2
    max_last = width - 2 if last_needs_tail else width - 1
    candidates = list(range(min_first, max_last + 1))
    valid: list[list[int]] = []

    def dfs(start_idx: int, chosen: list[int]) -> None:
        if len(chosen) == count:
            valid.append(chosen.copy())
            return
        remaining = count - len(chosen) - 1
        for idx in range(start_idx, len(candidates)):
            col = candidates[idx]
            min_possible_last = col + remaining * min_gap
            if min_possible_last > max_last:
                break
            chosen.append(col)
            next_col = col + min_gap
            next_idx = np.searchsorted(candidates, next_col)
            dfs(int(next_idx), chosen)
            chosen.pop()

    dfs(0, [])
    if not valid:
        return None
    return random.choice(valid)


def _pick_dye_sequence(
    count: int,
    dye_colors: list[int],
    *,
    seed_color: int,
    prefer_tail: int | None = None,
) -> list[int]:
    seq: list[int] = []
    previous = seed_color
    for idx in range(count):
        pool = [c for c in dye_colors if c != previous]
        if not pool:
            pool = dye_colors[:]
        if idx == count - 1 and prefer_tail is not None and prefer_tail in pool:
            color = prefer_tail
        else:
            color = random.choice(pool)
        seq.append(color)
        previous = color
    return seq


def _expected_from_spec(
    grid: np.ndarray,
    legend_colors: set[int],
    row_specs: dict[int, dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    h, w = grid.shape
    assert h >= 3 and w >= 3
    for row, spec in row_specs.items():
        beam_color = int(spec["seed_color"])
        events = spec["events"]
        event_map = {int(c): int(color) for c, color in events}
        c = 1
        while c < w:
            if c in event_map:
                color = event_map[c]
                if color in legend_colors:
                    break
                beam_color = color
            else:
                out[row, c] = beam_color
            c += 1
    return out


def _validate_scene(
    grid: np.ndarray,
    row_specs: dict[int, dict[str, object]],
    legend_colors: set[int],
    seed_colors: list[int],
) -> bool:
    try:
        bg, parsed_legend, seeds, obstacles = _parse_scene(grid)
    except AssertionError:
        return False

    if bg != BG or parsed_legend != legend_colors:
        return False
    if [color for _, color in seeds] != seed_colors:
        return False
    if set(obstacles) != set(row_specs):
        return False

    wall_colors_seen = set()
    dye_count = 0
    boundary_rows = 0
    multidye_rows = 0

    for row, spec in row_specs.items():
        events = list(spec["events"])
        actual = sorted(obstacles.get(row, {}).items())
        if actual != events:
            return False

        walls = [color for _, color in events if color in legend_colors]
        dyes = [color for _, color in events if color not in legend_colors]
        if walls:
            wall_colors_seen.update(walls)
        else:
            boundary_rows += 1
        dye_count += len(dyes)
        if len(dyes) >= 2:
            multidye_rows += 1

        # Every dye must have visible painted cells after it before the next event or boundary.
        for idx, (col, color) in enumerate(events):
            if color in legend_colors:
                continue
            next_col = events[idx + 1][0] if idx + 1 < len(events) else grid.shape[1]
            if next_col - col <= 1:
                return False

    if wall_colors_seen != legend_colors:
        return False
    if dye_count < 3 or boundary_rows != 1 or multidye_rows < 1:
        return False

    expected = _expected_from_spec(grid, legend_colors, row_specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False

    # Semantic spot checks on each planned row.
    for row, spec in row_specs.items():
        seed_color = int(spec["seed_color"])
        events = list(spec["events"])
        current = seed_color
        start = 1
        for col, color in events:
            if col > start:
                if not np.all(actual[row, start:col] == current):
                    return False
            if color in legend_colors:
                if not np.array_equal(actual[row, col:], grid[row, col:]):
                    return False
                break
            current = color
            start = col + 1
        else:
            if start < grid.shape[1] and not np.all(actual[row, start:] == current):
                return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        seed_count = random.randint(3, 4)
        legend_count = random.randint(2, min(3, seed_count - 1))
        max_dye_count = 9 - seed_count - legend_count
        dye_count = random.randint(2, min(3, max_dye_count))

        pool = random.sample(PALETTE, seed_count + legend_count + dye_count)
        legend_colors = pool[:legend_count]
        seed_colors = pool[legend_count : legend_count + seed_count]
        dye_colors = pool[legend_count + seed_count :]

        row_patterns = ["wall_only", "dye_boundary", "multi_dye_wall"]
        if seed_count == 4:
            row_patterns.append("single_dye_wall")
        random.shuffle(row_patterns)

        min_width = 15 if "single_dye_wall" in row_patterns else 13
        width = random.randint(min_width, 24)
        height = random.randint(seed_count + 4, 18)
        if height > 30 or width > 30:
            continue

        available_rows = list(range(2, height))
        seed_rows = sorted(random.sample(available_rows, seed_count))

        grid = np.full((height, width), BG, dtype=int)
        legend_start = random.randint(0, width - legend_count)
        grid[0, legend_start : legend_start + legend_count] = legend_colors

        wall_rows = [row for row, pattern in zip(seed_rows, row_patterns) if pattern != "dye_boundary"]
        if len(wall_rows) < legend_count:
            continue
        wall_assignments = {}
        shuffled_rows = wall_rows[:]
        random.shuffle(shuffled_rows)
        for row, color in zip(shuffled_rows, legend_colors):
            wall_assignments[row] = color
        for row in wall_rows:
            wall_assignments.setdefault(row, random.choice(legend_colors))

        common_tail = random.choice(dye_colors) if random.random() < 0.6 else None
        row_specs: dict[int, dict[str, object]] = {}

        ok = True
        for idx, (row, seed_color, pattern) in enumerate(zip(seed_rows, seed_colors, row_patterns)):
            grid[row, 0] = seed_color

            if pattern == "wall_only":
                event_colors = [wall_assignments[row]]
            elif pattern == "single_dye_wall":
                dye = _pick_dye_sequence(1, dye_colors, seed_color=seed_color, prefer_tail=common_tail)[0]
                event_colors = [dye, wall_assignments[row]]
            elif pattern == "multi_dye_wall":
                dye_len = random.randint(2, 3)
                dyes = _pick_dye_sequence(dye_len, dye_colors, seed_color=seed_color, prefer_tail=common_tail)
                event_colors = dyes + [wall_assignments[row]]
            elif pattern == "dye_boundary":
                dye_len = random.randint(1, 3)
                event_colors = _pick_dye_sequence(dye_len, dye_colors, seed_color=seed_color, prefer_tail=common_tail)
            else:
                ok = False
                break

            positions = _sample_positions(width, len(event_colors), last_needs_tail=event_colors[-1] not in legend_colors)
            if positions is None:
                ok = False
                break

            events = list(zip(positions, event_colors))
            for col, color in events:
                grid[row, col] = color
            row_specs[row] = {"seed_color": seed_color, "pattern": pattern, "events": events}

        if not ok:
            continue

        if not _validate_scene(grid, row_specs, set(legend_colors), seed_colors):
            continue
        return grid

    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
