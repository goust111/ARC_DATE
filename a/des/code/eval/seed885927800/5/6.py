import random
from collections import Counter

import numpy as np


BG_COLOR = 0
BASELINE_COLOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 500


def _baseline_color(grid: np.ndarray) -> int:
    row = grid[-1]
    assert bool((row == row[0]).all())
    return int(row[0])


def _background_color(grid: np.ndarray, baseline: int) -> int:
    counts = Counter(map(int, grid[:-1].ravel()))
    counts.pop(int(baseline), None)
    if counts:
        return int(counts.most_common(1)[0][0])
    return BG_COLOR


def _priority_sequence(grid: np.ndarray, bg: int, baseline: int) -> list[int]:
    seq = []
    for r in range(int(grid.shape[0] - 1)):
        color = int(grid[r, 0])
        if color not in (int(bg), int(baseline)):
            seq.append(int(color))
    return seq


def _extract_towers(grid: np.ndarray, bg: int, baseline: int) -> list[dict]:
    h, w = grid.shape
    towers = []
    for c in range(1, int(w)):
        body = int(grid[h - 2, c])
        if body in (int(bg), int(baseline)):
            continue

        r = h - 2
        while r >= 0 and int(grid[r, c]) == int(body):
            r -= 1
        top_r = int(r + 1)
        height = int(h - 1 - top_r)
        if height < 2:
            continue

        marker_r = int(top_r - 1)
        if marker_r < 0:
            continue
        marker = int(grid[marker_r, c])
        if marker in (int(bg), int(baseline), int(body)):
            continue
        if marker_r > 0 and bool((grid[:marker_r, c] != int(bg)).any()):
            continue

        towers.append(
            {
                "body": int(body),
                "marker": int(marker),
                "col": int(c),
                "top_r": int(top_r),
                "height": int(height),
            }
        )

    towers.sort(key=lambda tower: tower["col"])
    assert towers
    assert len({tower["body"] for tower in towers}) == len(towers)
    assert len({tower["marker"] for tower in towers}) == len(towers)
    return towers


def _half_diamond_cells(top_r: int, col: int, radius: int, h: int, w: int) -> set[tuple[int, int]]:
    cells = set()
    for dr in range(int(radius) + 1):
        r = int(top_r - dr)
        if r < 0:
            break
        span = int(radius - dr)
        c0 = max(0, int(col - span))
        c1 = min(int(w - 1), int(col + span))
        for c in range(int(c0), int(c1) + 1):
            cells.add((int(r), int(c)))
    return cells


def _tower_coverages(grid: np.ndarray, towers: list[dict]) -> list[set[tuple[int, int]]]:
    h, w = grid.shape
    bg = _background_color(grid, _baseline_color(grid))
    coverages = []
    for tower in towers:
        coverages.append(
            {
                cell
                for cell in _half_diamond_cells(
                    int(tower["top_r"]),
                    int(tower["col"]),
                    int(tower["height"]),
                    int(h),
                    int(w),
                )
                if int(grid[cell]) == int(bg)
            }
        )
    return coverages


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    baseline = _baseline_color(g)
    bg = _background_color(g, baseline)
    towers = _extract_towers(g, bg, baseline)
    priorities = _priority_sequence(g, bg, baseline)

    markers = [tower["marker"] for tower in towers]
    assert len(priorities) == len(towers)
    assert set(priorities) == set(markers)
    rank = {color: idx for idx, color in enumerate(priorities)}

    out = g.copy()
    out[:-1, 0] = int(bg)
    original_bg = g == int(bg)

    for tower in sorted(towers, key=lambda item: rank[int(item["marker"])], reverse=True):
        top_r = int(tower["top_r"])
        col = int(tower["col"])
        radius = int(tower["height"])
        for dr in range(int(radius) + 1):
            r = int(top_r - dr)
            if r < 0:
                break
            span = int(radius - dr)
            c0 = max(0, int(col - span))
            c1 = min(int(w - 1), int(col + span))
            mask = original_bg[r, c0 : c1 + 1]
            if bool(mask.any()):
                cols = np.nonzero(mask)[0] + int(c0)
                out[r, cols] = int(tower["body"])

    return out


def _sample_spaced(count: int, low: int, high: int, min_gap: int, rng: random.Random) -> list[int] | None:
    if count == 0:
        return []
    if low > high:
        return None
    if int(high - low) < int((count - 1) * min_gap):
        return None
    values = list(range(int(low), int(high) + 1))
    for _ in range(300):
        picks = sorted(rng.sample(values, int(count)))
        if all(int(b - a) >= int(min_gap) for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _priority_order_ok(order: list[int], cols: list[int], heights: list[int]) -> bool:
    if len(order) < 3:
        return True
    left_to_right = sorted(range(len(order)), key=lambda idx: cols[idx])
    tall_to_short = sorted(range(len(order)), key=lambda idx: (-heights[idx], cols[idx]))
    return order != left_to_right and order != tall_to_short


def _branch_ok(
    grid: np.ndarray,
    towers: list[dict],
    want_overlap: bool,
) -> bool:
    coverages = _tower_coverages(grid, towers)
    has_overlap = False
    for i in range(len(coverages)):
        for j in range(i + 1, len(coverages)):
            if coverages[i] & coverages[j]:
                has_overlap = True
                break
        if has_overlap:
            break
    return has_overlap == bool(want_overlap)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    tower_count = int(rng.randint(2, 4))
    want_overlap = bool(rng.randrange(2))
    heights = [int(rng.randint(2, 8)) for _ in range(tower_count)]
    max_height = max(heights)

    h_low = max(int(max_height + 2), int(2 * tower_count + 1))
    h_high = int(h_low + 5)
    grid_h = int(rng.randint(h_low, h_high))

    priority_rows = _sample_spaced(tower_count, 0, int(grid_h - 2), 2, rng)
    if priority_rows is None:
        return None

    palette = list(COLOR_POOL)
    rng.shuffle(palette)
    body_colors = [int(color) for color in palette[:tower_count]]
    marker_colors = [int(color) for color in palette[tower_count : 2 * tower_count]]

    left_bound = int(max_height + 1)
    min_width = max(int(left_bound + 3 * (tower_count - 1) + 2), 12)
    extra_width = 6 if want_overlap else int(max_height + 8)
    grid_w = int(rng.randint(min_width, min_width + extra_width))

    for _ in range(300):
        cols = _sample_spaced(tower_count, left_bound, int(grid_w - 2), 3, rng)
        if cols is None:
            return None

        priority_order = list(range(tower_count))
        rng.shuffle(priority_order)
        if not _priority_order_ok(priority_order, cols, heights):
            continue

        grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
        grid[-1, :] = BASELINE_COLOR

        for idx, (col, height) in enumerate(zip(cols, heights)):
            top_r = int(grid_h - 1 - height)
            grid[top_r : grid_h - 1, int(col)] = int(body_colors[idx])
            grid[top_r - 1, int(col)] = int(marker_colors[idx])

        for row, idx in zip(priority_rows, priority_order):
            grid[int(row), 0] = int(marker_colors[idx])

        towers = _extract_towers(grid, BG_COLOR, BASELINE_COLOR)
        if len(towers) != tower_count:
            continue
        if set(_priority_sequence(grid, BG_COLOR, BASELINE_COLOR)) != set(marker_colors):
            continue
        if not _branch_ok(grid, towers, want_overlap):
            continue

        out = main(grid)
        if out.shape != grid.shape:
            continue
        if not bool((out[:-1, 0] == BG_COLOR).all()):
            continue
        if not bool((out[-1, :] == BASELINE_COLOR).all()):
            continue
        if not bool(np.any(out != grid)):
            continue

        return grid

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
