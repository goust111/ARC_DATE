from __future__ import annotations

import numpy as np


BG = 0
TERRAIN = 5
TOWER_PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
GEN_RETRY_LIMIT = 500


def _background_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _terrain_color_and_rows(grid: np.ndarray, bg: int) -> tuple[int, np.ndarray]:
    candidates: list[int] = []
    rows_by_color: dict[int, np.ndarray] = {}
    for color_raw in np.unique(grid):
        color = int(color_raw)
        if color == bg:
            continue
        mask = grid == color
        if np.all(mask.sum(axis=0) == 1):
            candidates.append(color)
            rows_by_color[color] = np.argmax(mask, axis=0).astype(int)
    assert len(candidates) == 1, "expected a unique terrain contour color"
    terrain = candidates[0]
    return terrain, rows_by_color[terrain]


def _extract_towers(
    grid: np.ndarray, bg: int, terrain: int, terrain_rows: np.ndarray
) -> list[dict[str, int]]:
    h, w = grid.shape
    tower_mask = np.zeros((h, w), dtype=bool)
    towers: list[dict[str, int]] = []

    for c in range(w):
        terrain_r = int(terrain_rows[c])
        r = terrain_r - 1
        if r >= 0 and int(grid[r, c]) not in (bg, terrain):
            color = int(grid[r, c])
            height = 0
            while r >= 0 and int(grid[r, c]) == color:
                tower_mask[r, c] = True
                height += 1
                r -= 1
            if r >= 0:
                assert np.all(grid[: r + 1, c] == bg), "tower must be contiguous"
            towers.append({"col": int(c), "height": int(height), "color": int(color)})
        else:
            if terrain_r > 0:
                assert np.all(grid[:terrain_r, c] == bg), "only towers may sit above terrain"

    unexplained = (grid != bg) & (grid != terrain) & (~tower_mask)
    assert not bool(unexplained.any()), "all foreground pixels above terrain must be towers"

    cols = [tower["col"] for tower in towers]
    assert all(b - a >= 2 for a, b in zip(cols, cols[1:])), "towers must have a gap"
    return towers


def _shadow_winners(width: int, towers: list[dict[str, int]]) -> tuple[list[int], list[int]]:
    winner_colors = [-1] * width
    winner_idx = [-1] * width
    winner_priority = [(-1, -1)] * width

    for idx, tower in enumerate(towers):
        col = int(tower["col"])
        height = int(tower["height"])
        color = int(tower["color"])
        assert col + height < width, "tower shadow must fit inside the grid"
        priority = (height, col)
        for target_col in range(col + 1, col + height + 1):
            if priority > winner_priority[target_col]:
                winner_priority[target_col] = priority
                winner_colors[target_col] = color
                winner_idx[target_col] = idx

    return winner_colors, winner_idx


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project colored tower shadows onto an irregular terrain contour.

    Input structure
    - Background is black.
    - One gray terrain pixel appears in every column, forming a bottom contour.
    - Some columns contain a single-color vertical tower rooted directly above the terrain.

    Rule
    - A tower of height h paints h terrain columns to its right with its own color.
    - Overlaps resolve by higher tower first; equal heights resolve by the rightmost tower.
    - Towers disappear in the output; only the terrain line remains.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _background_color(g)
    terrain, terrain_rows = _terrain_color_and_rows(g, bg)
    towers = _extract_towers(g, bg, terrain, terrain_rows)

    out = np.full_like(g, bg)
    cols = np.arange(g.shape[1])
    out[terrain_rows, cols] = terrain

    winner_colors, _ = _shadow_winners(g.shape[1], towers)
    for c, color in enumerate(winner_colors):
        if color != -1:
            out[int(terrain_rows[c]), int(c)] = int(color)

    return out


def _random_terrain_rows(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    row_lo = max(4, height - 7)
    row_hi = height - 2
    rows = np.empty(width, dtype=int)
    rows[0] = int(rng.integers(row_lo, row_hi + 1))
    steps = np.array([-2, -1, 0, 1, 2], dtype=int)
    probs = np.array([0.08, 0.24, 0.36, 0.24, 0.08], dtype=float)
    for c in range(1, width):
        step = int(rng.choice(steps, p=probs))
        rows[c] = int(np.clip(rows[c - 1] + step, row_lo, row_hi))
    if int(rows.max() - rows.min()) < 2:
        pivot = int(rng.integers(0, width))
        delta = 2 if rows[pivot] <= (row_lo + row_hi) // 2 else -2
        rows[pivot] = int(np.clip(rows[pivot] + delta, row_lo, row_hi))
    return rows


def _sample_spaced_columns(
    rng: np.random.Generator, width: int, count: int
) -> list[int]:
    usable = width - 1
    assert 1 <= count <= (usable + 1) // 2
    compressed = np.sort(rng.choice(usable - count + 1, size=count, replace=False))
    return [int(value + idx) for idx, value in enumerate(compressed)]


def _baseline_heights(
    rng: np.random.Generator, cols: list[int], max_heights: list[int]
) -> list[int]:
    heights: list[int] = []
    for idx, max_height in enumerate(max_heights):
        limit = int(max_height)
        if idx + 1 < len(cols):
            limit = min(limit, cols[idx + 1] - cols[idx])
        heights.append(int(rng.integers(1, limit + 1)))
    if max(heights) == 1:
        candidates = [idx for idx, limit in enumerate(max_heights) if limit >= 2]
        if candidates:
            chosen = int(rng.choice(candidates))
            upper = int(max_heights[chosen])
            if chosen + 1 < len(cols):
                upper = min(upper, cols[chosen + 1] - cols[chosen])
            heights[chosen] = int(rng.integers(2, upper + 1))
    return heights


def _overlap_span(left: dict[str, int], right: dict[str, int]) -> tuple[int, int] | None:
    start = max(int(left["col"]) + 1, int(right["col"]) + 1)
    end = min(int(left["col"]) + int(left["height"]), int(right["col"]) + int(right["height"]))
    if start > end:
        return None
    return int(start), int(end)


def _has_overlap_of_kind(
    towers: list[dict[str, int]], same_height: bool | None
) -> bool:
    for i, left in enumerate(towers):
        for right in towers[i + 1 :]:
            if _overlap_span(left, right) is None:
                continue
            if same_height is None:
                return True
            if (int(left["height"]) == int(right["height"])) == same_height:
                return True
    return False


def _render_input(
    height: int, width: int, terrain_rows: np.ndarray, towers: list[dict[str, int]]
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for c, terrain_r in enumerate(terrain_rows):
        grid[int(terrain_r), int(c)] = TERRAIN
    for tower in towers:
        col = int(tower["col"])
        tower_height = int(tower["height"])
        color = int(tower["color"])
        terrain_r = int(terrain_rows[col])
        grid[terrain_r - tower_height : terrain_r, col] = color
    return grid


def _render_expected_output(
    height: int, width: int, terrain_rows: np.ndarray, towers: list[dict[str, int]]
) -> np.ndarray:
    out = np.full((height, width), BG, dtype=int)
    for c, terrain_r in enumerate(terrain_rows):
        out[int(terrain_r), int(c)] = TERRAIN
    winner_colors, _ = _shadow_winners(width, towers)
    for c, color in enumerate(winner_colors):
        if color != -1:
            out[int(terrain_rows[c]), int(c)] = int(color)
    return out


def _pair_visible(
    towers: list[dict[str, int]], pair_idx: int, winner_idx: list[int], expect_right: bool
) -> bool:
    left = towers[pair_idx]
    right = towers[pair_idx + 1]
    span = _overlap_span(left, right)
    if span is None:
        return False
    start, end = span
    winner = pair_idx + 1 if expect_right else pair_idx
    return any(winner_idx[col] == winner for col in range(start, end + 1))


def _pick_towers(
    rng: np.random.Generator, terrain_rows: np.ndarray, width: int, scenario: str
) -> list[dict[str, int]] | None:
    max_towers = min(len(TOWER_PALETTE), 5, width // 2)
    if max_towers < 2:
        return None

    if scenario == "non_overlap":
        count = int(rng.integers(2, max_towers + 1))
    else:
        count = int(rng.integers(2, min(4, max_towers) + 1))

    for _ in range(200):
        cols = _sample_spaced_columns(rng, width, count)
        max_heights = [int(min(7, int(terrain_rows[col]), width - 1 - col)) for col in cols]
        if min(max_heights) < 1:
            continue

        heights = _baseline_heights(rng, cols, max_heights)
        pair_idx: int | None = None

        if scenario == "overlap_equal":
            feasible = [
                idx
                for idx in range(count - 1)
                if min(max_heights[idx], max_heights[idx + 1]) >= cols[idx + 1] - cols[idx] + 1
            ]
            if not feasible:
                continue
            pair_idx = int(rng.choice(feasible))
            low = cols[pair_idx + 1] - cols[pair_idx] + 1
            high = min(max_heights[pair_idx], max_heights[pair_idx + 1])
            shared = int(rng.integers(low, high + 1))
            heights[pair_idx] = shared
            heights[pair_idx + 1] = shared
        elif scenario == "overlap_diff":
            feasible = [
                idx
                for idx in range(count - 1)
                if max_heights[idx] >= cols[idx + 1] - cols[idx] + 1 and max_heights[idx + 1] >= 2
            ]
            if not feasible:
                continue
            pair_idx = int(rng.choice(feasible))
            low = cols[pair_idx + 1] - cols[pair_idx] + 1
            heights[pair_idx] = int(rng.integers(low, max_heights[pair_idx] + 1))
            choices = [value for value in range(1, max_heights[pair_idx + 1] + 1) if value != heights[pair_idx]]
            if not choices:
                continue
            heights[pair_idx + 1] = int(rng.choice(np.array(choices, dtype=int)))

        colors = rng.choice(TOWER_PALETTE, size=count, replace=False).astype(int)
        towers = [
            {"col": int(col), "height": int(height), "color": int(color)}
            for col, height, color in zip(cols, heights, colors)
        ]

        winner_colors, winner_idx = _shadow_winners(width, towers)
        visible_colors = {int(color) for color in winner_colors if color != -1}
        if len(visible_colors) < 2:
            continue
        if winner_colors.count(-1) == 0:
            continue

        if scenario == "non_overlap":
            if _has_overlap_of_kind(towers, None):
                continue
        elif scenario == "overlap_equal":
            if pair_idx is None or not _has_overlap_of_kind(towers, True):
                continue
            if not _pair_visible(towers, pair_idx, winner_idx, expect_right=True):
                continue
        else:
            if pair_idx is None or not _has_overlap_of_kind(towers, False):
                continue
            left = towers[pair_idx]
            right = towers[pair_idx + 1]
            expect_right = int(right["height"]) > int(left["height"])
            if not _pair_visible(towers, pair_idx, winner_idx, expect_right=expect_right):
                continue

        return towers

    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(10, 19))
    width = int(rng.integers(12, 23))
    terrain_rows = _random_terrain_rows(rng, height, width)
    scenario = str(rng.choice(np.array(["non_overlap", "overlap_diff", "overlap_equal"])))
    towers = _pick_towers(rng, terrain_rows, width, scenario)
    if towers is None:
        return None

    grid = _render_input(height, width, terrain_rows, towers)
    expected = _render_expected_output(height, width, terrain_rows, towers)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
