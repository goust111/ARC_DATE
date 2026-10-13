from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
RECEIVER_COLOR = 5
ACTIVE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# Generator configuration.
GRID_HEIGHT_RANGE = (13, 20)
GRID_WIDTH_RANGE = (15, 24)
TOWER_COUNT_RANGE = (3, 5)
TOWER_HEIGHT_RANGE = (1, 5)
GENERATION_ATTEMPTS = 400


def _non_bg_runs(row: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(row):
        if int(value) != BG_COLOR:
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(row) - 1))
    return runs


def _extract_legend(grid: np.ndarray) -> tuple[list[int], int]:
    bottom = np.asarray(grid[-1], dtype=int)
    runs = _non_bg_runs(bottom)
    assert runs, "expected a legend on the last row"
    start, end = max(runs, key=lambda ab: ab[1] - ab[0])
    colors = [int(x) for x in bottom[start : end + 1]]
    assert len(colors) == len(set(colors)), "legend colors must be distinct"
    return colors, int(start)


def _extract_towers(grid: np.ndarray) -> list[dict[str, int]]:
    scene = np.asarray(grid[:-1], dtype=int)
    h, w = scene.shape
    towers: list[dict[str, int]] = []
    for c in range(w):
        r = 0
        while r < h:
            color = int(scene[r, c])
            if color in (BG_COLOR, RECEIVER_COLOR):
                r += 1
                continue
            top = r
            while r + 1 < h and int(scene[r + 1, c]) == color:
                r += 1
            bottom = r
            towers.append(
                {
                    "top": int(top),
                    "col": int(c),
                    "height": int(bottom - top + 1),
                    "color": int(color),
                }
            )
            r += 1
    return towers


def _coverage_map(shape: tuple[int, int], towers: list[dict[str, int]]) -> np.ndarray:
    h, w = shape
    counts = np.zeros((h, w), dtype=int)
    for tower in towers:
        top = int(tower["top"])
        col = int(tower["col"])
        radius = int(tower["height"])
        for dc in range(-radius, radius + 1):
            cc = col + dc
            if 0 <= cc < w:
                counts[top, cc] += 1
        for dr in range(-radius, radius + 1):
            rr = top + dr
            if 0 <= rr < h:
                counts[rr, col] += 1
        counts[top, col] -= 1
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey receiver by the number of signal-tower crosses that cover it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend_colors, _legend_start = _extract_legend(g)
    towers = _extract_towers(g)
    counts = _coverage_map((g.shape[0] - 1, g.shape[1]), towers)

    out = np.array(g, copy=True)
    receiver_positions = np.argwhere(g[:-1] == RECEIVER_COLOR)
    for r, c in receiver_positions:
        n_cover = int(counts[int(r), int(c)])
        if 1 <= n_cover <= len(legend_colors):
            out[int(r), int(c)] = int(legend_colors[n_cover - 1])
    return out


def _build_shared_row_towers(scene_h: int, width: int, heights: list[int]) -> list[dict[str, int]] | None:
    heights = heights[:]
    random.shuffle(heights)
    max_h = max(heights)
    if scene_h - 1 - max_h < max_h:
        return None

    rel_cols = [0]
    for prev_h, cur_h in zip(heights, heights[1:]):
        max_gap = min(4, prev_h + cur_h)
        if max_gap < 2:
            return None
        rel_cols.append(rel_cols[-1] + random.randint(2, max_gap))

    shift_lo = max(h - rel for h, rel in zip(heights, rel_cols))
    shift_hi = min((width - 1 - h) - rel for h, rel in zip(heights, rel_cols))
    if shift_lo > shift_hi:
        return None

    top = random.randint(max_h, scene_h - 1 - max_h)
    shift = random.randint(shift_lo, shift_hi)
    cols = [shift + rel for rel in rel_cols]

    towers: list[dict[str, int]] = []
    for h, c in zip(heights, cols):
        towers.append({"top": int(top), "col": int(c), "height": int(h)})
    return towers


def _perturb_towers(towers: list[dict[str, int]], scene_h: int) -> list[dict[str, int]]:
    perturbed: list[dict[str, int]] = []
    for tower in towers:
        height = int(tower["height"])
        top = int(tower["top"])
        low = height
        high = scene_h - 1 - height
        delta_choices = [0]
        for delta in (-2, -1, 1, 2):
            new_top = top + delta
            if low <= new_top <= high and abs(delta) <= height:
                delta_choices.append(delta)
        new_top = top + random.choice(delta_choices)
        perturbed.append({"top": int(new_top), "col": int(tower["col"]), "height": int(height)})
    return perturbed


def _pick_receivers(
    by_count: dict[int, list[tuple[int, int]]],
    required_counts: list[int],
) -> list[tuple[int, int, int]] | None:
    occupied: list[tuple[int, int]] = []
    placements: list[tuple[int, int, int]] = []

    def can_use(cell: tuple[int, int]) -> bool:
        r, c = cell
        return all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in occupied)

    for count in required_counts:
        candidates = by_count.get(count, [])[:]
        random.shuffle(candidates)
        chosen = None
        for cell in candidates:
            if can_use(cell):
                chosen = cell
                break
        if chosen is None:
            for cell in candidates:
                if cell not in occupied:
                    chosen = cell
                    break
        if chosen is None:
            return None
        occupied.append(chosen)
        placements.append((int(chosen[0]), int(chosen[1]), int(count)))
    return placements


def _expected_output(
    grid: np.ndarray,
    receivers: list[tuple[int, int, int]],
    legend_colors: list[int],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for r, c, count in receivers:
        if count > 0:
            out[r, c] = int(legend_colors[count - 1])
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    scene_h = h - 1

    tower_count = random.randint(*TOWER_COUNT_RANGE)
    heights = [random.randint(*TOWER_HEIGHT_RANGE) for _ in range(tower_count)]
    towers = _build_shared_row_towers(scene_h, w, heights)
    if towers is None:
        return None
    if random.random() < 0.45:
        towers = _perturb_towers(towers, scene_h)

    grid = np.full((h, w), BG_COLOR, dtype=int)

    tower_colors = random.sample(ACTIVE_COLORS, tower_count)
    for tower, color in zip(towers, tower_colors):
        tower["color"] = int(color)
        top = int(tower["top"])
        col = int(tower["col"])
        height = int(tower["height"])
        grid[top : top + height, col] = int(color)

    counts = _coverage_map((scene_h, w), towers)
    occupied = grid[:-1] != BG_COLOR
    by_count: dict[int, list[tuple[int, int]]] = {}
    for r in range(scene_h):
        for c in range(w):
            if occupied[r, c]:
                continue
            by_count.setdefault(int(counts[r, c]), []).append((int(r), int(c)))

    positive_counts = [count for count in sorted(by_count) if count > 0]
    if 0 not in by_count or len(by_count[0]) < 2:
        return None
    if len(positive_counts) < 2:
        return None

    max_mapped = min(5, positive_counts[-1])
    legend_len = random.randint(2, max_mapped)
    usable_counts = [count for count in positive_counts if count <= legend_len]
    if len(usable_counts) < 2:
        return None

    if tower_count + legend_len <= len(ACTIVE_COLORS):
        remaining = [color for color in ACTIVE_COLORS if color not in tower_colors]
        legend_colors = random.sample(remaining, legend_len)
    else:
        legend_colors = random.sample(ACTIVE_COLORS, legend_len)

    legend_start = random.randint(0, w - legend_len)
    grid[-1, legend_start : legend_start + legend_len] = np.array(legend_colors, dtype=int)

    positive_pick_count = random.randint(2, len(usable_counts))
    used_positive = sorted(random.sample(usable_counts, positive_pick_count))

    required_counts = [0, 0]
    for count in used_positive:
        copies = 1 + int(random.random() < 0.5)
        required_counts.extend([count] * copies)
    extra_count = random.choice([0] + used_positive)
    required_counts.append(int(extra_count))
    random.shuffle(required_counts)

    receivers = _pick_receivers(by_count, required_counts)
    if receivers is None:
        return None

    for r, c, _count in receivers:
        grid[r, c] = RECEIVER_COLOR

    expected = _expected_output(grid, receivers, legend_colors)
    if not np.array_equal(main(grid), expected):
        raise AssertionError("generator and solver disagree")

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower puzzle")
