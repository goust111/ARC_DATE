from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


SKY = 8
TERRAIN = 3
TOWER_COLORS = [0, 1, 2, 4, 5, 6, 7, 9]
MAX_ATTEMPTS = 400


@dataclass(frozen=True)
class Tower:
    col: int
    top_row: int
    body_height: int
    body_color: int
    signal_color: int


def _mode_color(values: np.ndarray) -> int:
    vals, counts = np.unique(values.astype(int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _terrain_tops(grid: np.ndarray, terrain_color: int) -> list[int]:
    height, width = grid.shape
    tops: list[int] = []
    for c in range(width):
        rows = np.flatnonzero(grid[:, c] == terrain_color)
        assert rows.size > 0, "every column must contain terrain"
        top = int(rows[0])
        assert int(rows[-1]) == height - 1, "terrain must reach the bottom edge"
        assert np.array_equal(rows, np.arange(top, height)), "terrain must be contiguous upward from the bottom"
        tops.append(top)
    return tops


def _extract_scene(grid: np.ndarray) -> tuple[int, int, list[int], list[Tower]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sky_color = _mode_color(g[0])
    terrain_color = _mode_color(g[-1])
    assert sky_color != terrain_color, "sky and terrain colors must differ"

    terrain_tops = _terrain_tops(g, terrain_color)
    towers: list[Tower] = []
    for c, terrain_top in enumerate(terrain_tops):
        if terrain_top == 0:
            continue
        above = g[:terrain_top, c]
        solid_rows = np.flatnonzero(above != sky_color)
        if solid_rows.size == 0:
            continue

        start = int(solid_rows[0])
        expected = np.arange(start, terrain_top)
        assert np.array_equal(solid_rows, expected), "tower marker and body must form one contiguous vertical segment"
        assert terrain_top - start >= 2, "tower column must contain a marker and at least one body cell"

        signal_row = start
        top_row = start + 1
        body_rows = np.arange(top_row, terrain_top)
        body_colors = np.unique(g[body_rows, c])
        assert body_colors.size == 1, "tower body must be monochrome"

        signal_color = int(g[signal_row, c])
        body_color = int(body_colors[0])
        assert signal_color not in (sky_color, terrain_color)
        assert body_color not in (sky_color, terrain_color)
        assert signal_color != body_color

        towers.append(
            Tower(
                col=c,
                top_row=top_row,
                body_height=terrain_top - top_row,
                body_color=body_color,
                signal_color=signal_color,
            )
        )

    assert towers, "expected at least one tower"
    return sky_color, terrain_color, terrain_tops, towers


def _compute_claims(
    grid: np.ndarray,
    sky_color: int,
    terrain_tops: list[int],
    towers: list[Tower],
) -> dict[tuple[int, int], list[int]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    claims: dict[tuple[int, int], list[int]] = {}

    for tower_idx, tower in enumerate(towers):
        for step in (-1, 1):
            c = tower.col + step
            while 0 <= c < width:
                if terrain_tops[c] <= tower.top_row:
                    break
                for r in range(tower.top_row, terrain_tops[c]):
                    if 0 <= r < height and int(g[r, c]) == sky_color:
                        claims.setdefault((r, c), []).append(tower_idx)
                c += step
    return claims


def _pick_winner(cell: tuple[int, int], tower_ids: list[int], towers: list[Tower]) -> int:
    _r, c = cell
    return min(tower_ids, key=lambda idx: (abs(c - towers[idx].col), towers[idx].top_row, towers[idx].col))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project colored signal shadows from towers across a terrain silhouette, with mountains blocking the signal.

    Input structure:
    - Azure sky fills the background.
    - Each column has a bottom-anchored green terrain stack, creating hills and valleys.
    - Some columns have a vertical tower above the terrain in one color, plus a 1x1 signal-color pixel directly above the tower top.

    Rule:
    - Each tower emits horizontally left and right from its top row.
    - In every reachable column, recolor the azure cells from the tower-top row down to just above that column's terrain top.
    - A column whose terrain reaches the tower-top height blocks the signal there and farther in that direction.
    - Overlaps are resolved by smaller horizontal distance to the tower; ties go to the higher tower.

    Key insight:
    - The lone pixel above each tower is the emitted signal color, while the tower body only marks the emitter location and height.
    """
    g = np.asarray(grid, dtype=int)
    sky_color, _terrain_color, terrain_tops, towers = _extract_scene(g)
    claims = _compute_claims(g, sky_color, terrain_tops, towers)

    out = g.copy()
    for cell, tower_ids in claims.items():
        winner = _pick_winner(cell, tower_ids, towers)
        out[cell] = towers[winner].signal_color
    return out


def _random_walk_heights(height: int, width: int) -> list[int]:
    terrain_max = height - 2
    low = 2
    high = min(6, terrain_max - 1)
    start = random.randint(low, high)
    heights = [start]
    for _ in range(1, width):
        step = random.choice([-2, -1, 0, 1, 2])
        heights.append(max(low, min(terrain_max, heights[-1] + step)))
    if max(heights) - min(heights) < 2:
        pivot = random.randrange(width)
        heights[pivot] = min(terrain_max, heights[pivot] + 2)
    return heights


def _sample_colors(count: int) -> list[tuple[int, int]]:
    colors = random.sample(TOWER_COLORS, 2 * count)
    return [(colors[2 * i], colors[2 * i + 1]) for i in range(count)]


def _render_input(height: int, width: int, terrain_heights: list[int], towers: list[tuple[int, int, int, int]]) -> np.ndarray:
    grid = np.full((height, width), SKY, dtype=int)
    for c, terrain_height in enumerate(terrain_heights):
        grid[height - terrain_height :, c] = TERRAIN

    for col, body_height, body_color, signal_color in towers:
        terrain_top = height - terrain_heights[col]
        top_row = terrain_top - body_height
        grid[top_row:terrain_top, col] = body_color
        grid[top_row - 1, col] = signal_color
    return grid


def _has_blocking(towers: list[Tower], terrain_tops: list[int], width: int) -> bool:
    for tower in towers:
        for step in (-1, 1):
            c = tower.col + step
            while 0 <= c < width:
                if terrain_tops[c] <= tower.top_row:
                    return True
                c += step
    return False


def _claim_features(
    claims: dict[tuple[int, int], list[int]],
    towers: list[Tower],
) -> tuple[bool, bool]:
    overlap_non_tie = False
    tie_by_height = False
    for (_r, c), tower_ids in claims.items():
        if len(tower_ids) < 2:
            continue
        dists = [abs(c - towers[idx].col) for idx in tower_ids]
        best_dist = min(dists)
        nearest = [idx for idx in tower_ids if abs(c - towers[idx].col) == best_dist]
        if len(nearest) == 1:
            overlap_non_tie = True
        else:
            top_rows = {towers[idx].top_row for idx in nearest}
            if len(top_rows) > 1:
                tie_by_height = True
    return overlap_non_tie, tie_by_height


def _sample_body_height(height: int, ground_height: int, minimum: int = 1) -> int | None:
    maximum = min(5, height - ground_height - 2)
    if maximum < minimum:
        return None
    return random.randint(minimum, maximum)


def _low_height(limit: int) -> int:
    return random.randint(2, max(2, limit))


def _blocker_height(height: int, altitude: int) -> int:
    terrain_max = height - 2
    return random.randint(altitude, terrain_max)


def _build_single_input() -> np.ndarray | None:
    height = random.randint(12, 18)
    width = random.randint(10, 20)
    terrain_heights = _random_walk_heights(height, width)

    col = random.randint(2, width - 3)
    terrain_heights[col] = random.randint(2, min(5, height - 6))
    body_height = _sample_body_height(height, terrain_heights[col], minimum=1)
    if body_height is None:
        return None
    altitude = terrain_heights[col] + body_height

    open_side = random.choice([-1, 1])
    for step in (-1, 1):
        span = col if step == -1 else width - 1 - col
        if span <= 0:
            continue
        if step == open_side:
            open_len = random.randint(1, min(4, span))
            for offset in range(1, open_len + 1):
                terrain_heights[col + step * offset] = _low_height(altitude - 1)
            if open_len < span:
                terrain_heights[col + step * (open_len + 1)] = _blocker_height(height, altitude)
        else:
            pre_open = 0 if span == 1 or random.random() < 0.5 else 1
            for offset in range(1, pre_open + 1):
                terrain_heights[col + step * offset] = _low_height(altitude - 1)
            blocker_col = col + step * (pre_open + 1)
            if 0 <= blocker_col < width:
                terrain_heights[blocker_col] = _blocker_height(height, altitude)

    (body_color, signal_color) = _sample_colors(1)[0]
    towers = [(col, body_height, body_color, signal_color)]
    return _render_input(height, width, terrain_heights, towers)


def _build_overlap_input() -> np.ndarray | None:
    height = random.randint(12, 18)
    width = random.randint(12, 20)
    terrain_heights = _random_walk_heights(height, width)

    gaps = [gap for gap in (3, 5, 7) if gap <= width - 5]
    if not gaps:
        return None
    gap = random.choice(gaps)
    left_col = random.randint(2, width - gap - 3)
    right_col = left_col + gap

    terrain_heights[left_col] = random.randint(2, min(4, height - 7))
    terrain_heights[right_col] = random.randint(2, min(4, height - 7))
    left_body = _sample_body_height(height, terrain_heights[left_col], minimum=2)
    right_body = _sample_body_height(height, terrain_heights[right_col], minimum=2)
    if left_body is None or right_body is None:
        return None

    left_altitude = terrain_heights[left_col] + left_body
    right_altitude = terrain_heights[right_col] + right_body
    valley_cap = min(left_altitude, right_altitude) - 1
    if valley_cap < 2:
        return None

    for c in range(left_col + 1, right_col):
        terrain_heights[c] = _low_height(min(4, valley_cap))
    terrain_heights[left_col - 1] = _blocker_height(height, left_altitude)
    terrain_heights[right_col + 1] = _blocker_height(height, right_altitude)

    colors = _sample_colors(2)
    towers = [
        (left_col, left_body, colors[0][0], colors[0][1]),
        (right_col, right_body, colors[1][0], colors[1][1]),
    ]
    return _render_input(height, width, terrain_heights, towers)


def _build_tie_input() -> np.ndarray | None:
    height = random.randint(12, 18)
    width = random.randint(12, 20)
    terrain_heights = _random_walk_heights(height, width)

    gaps = [gap for gap in (4, 6, 8) if gap <= width - 5]
    if not gaps:
        return None
    gap = random.choice(gaps)
    left_col = random.randint(2, width - gap - 3)
    right_col = left_col + gap

    left_ground = random.randint(2, min(4, height - 7))
    right_ground = random.randint(2, min(4, height - 7))
    terrain_heights[left_col] = left_ground
    terrain_heights[right_col] = right_ground

    right_body = _sample_body_height(height, right_ground, minimum=1)
    if right_body is None:
        return None
    low_altitude = right_ground + right_body

    high_min = max(low_altitude + 1, left_ground + 1)
    high_max = min(left_ground + 5, height - 2, low_altitude + 3)
    if high_min > high_max:
        return None
    high_altitude = random.randint(high_min, high_max)
    left_body = high_altitude - left_ground
    if not (1 <= left_body <= 5):
        return None

    valley_cap = low_altitude - 1
    if valley_cap < 2:
        return None
    for c in range(left_col + 1, right_col):
        terrain_heights[c] = _low_height(min(4, valley_cap))

    terrain_heights[left_col - 1] = _blocker_height(height, high_altitude)
    terrain_heights[right_col + 1] = _blocker_height(height, low_altitude)

    colors = _sample_colors(2)
    towers = [
        (left_col, left_body, colors[0][0], colors[0][1]),
        (right_col, right_body, colors[1][0], colors[1][1]),
    ]
    return _render_input(height, width, terrain_heights, towers)


def _scenario_matches(grid: np.ndarray, scenario: str) -> bool:
    sky_color, _terrain_color, terrain_tops, towers = _extract_scene(grid)
    claims = _compute_claims(grid, sky_color, terrain_tops, towers)
    overlap_non_tie, tie_by_height = _claim_features(claims, towers)
    has_overlap = any(len(ids) > 1 for ids in claims.values())
    has_changes = bool(np.any(main(grid) != grid))
    has_block = _has_blocking(towers, terrain_tops, grid.shape[1])

    if not has_changes or not has_block:
        return False
    if scenario == "single":
        return len(towers) == 1 and not has_overlap
    if scenario == "overlap":
        return len(towers) == 2 and overlap_non_tie
    if scenario == "tie":
        return len(towers) == 2 and tie_by_height
    return False


def generate_input() -> np.ndarray:
    builders = {
        "single": _build_single_input,
        "overlap": _build_overlap_input,
        "tie": _build_tie_input,
    }
    scenarios = ["single", "overlap", "tie"]
    scenario = random.choice(scenarios)
    builder = builders[scenario]

    for _ in range(MAX_ATTEMPTS):
        grid = builder()
        if grid is None:
            continue
        try:
            if _scenario_matches(grid, scenario):
                return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid signal-shadow input")
