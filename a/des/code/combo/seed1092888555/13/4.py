import random
from typing import List, Tuple

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Collect the endpoint marker of each edge path and sort markers by path length."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    records = _extract_paths(g)
    assert records, "expected at least one edge path"
    records.sort(key=lambda item: item[0])
    return np.array([[marker for _length, marker in records]], dtype=int)


def _extract_paths(grid: np.ndarray) -> List[Tuple[int, int]]:
    h, w = grid.shape
    starts: List[Tuple[int, int, int, int]] = []

    for c in range(w):
        if grid[0, c] != 0:
            starts.append((0, c, 1, 0))
        if grid[h - 1, c] != 0:
            starts.append((h - 1, c, -1, 0))
    for r in range(1, h - 1):
        if grid[r, 0] != 0:
            starts.append((r, 0, 0, 1))
        if grid[r, w - 1] != 0:
            starts.append((r, w - 1, 0, -1))

    seen_starts = set()
    records: List[Tuple[int, int]] = []
    for r, c, dr, dc in starts:
        if (r, c) in seen_starts:
            continue
        body = int(grid[r, c])
        rr, cc = r, c
        length = 1
        seen_starts.add((r, c))

        while True:
            nr, nc = rr + dr, cc + dc
            assert 0 <= nr < h and 0 <= nc < w, "path must terminate inside the grid"
            nxt = int(grid[nr, nc])
            assert nxt != 0, "path must end with a marker before background"
            length += 1
            if nxt == body:
                rr, cc = nr, nc
                continue

            marker = nxt
            after_r, after_c = nr + dr, nc + dc
            if 0 <= after_r < h and 0 <= after_c < w:
                assert int(grid[after_r, after_c]) == 0, "marker must be terminal"
            records.append((length, marker))
            break

    return records


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)

    top_count = random.randint(1, 2)
    bottom_count = random.randint(1, 2)
    left_count = random.randint(1, 2)
    right_count = random.randint(1, 2)
    total = top_count + bottom_count + left_count + right_count

    min_center_w = max(top_count + bottom_count + 4, 6)
    min_center_h = max(left_count + right_count + 4, 6)
    if w < min_center_w + 4 or h < min_center_h + 4:
        return None

    center_w = random.randint(min_center_w, w - 4)
    center_h = random.randint(min_center_h, h - 4)
    c_lo = random.randint(2, w - center_w - 2)
    c_hi = c_lo + center_w - 1
    r_lo = random.randint(2, h - center_h - 2)
    r_hi = r_lo + center_h - 1

    top_pool = list(range(2, r_lo + 1))
    bottom_pool = list(range(2, h - r_hi))
    left_pool = list(range(2, c_lo + 1))
    right_pool = list(range(2, w - c_hi))
    if min(map(len, [top_pool, bottom_pool, left_pool, right_pool])) == 0:
        return None

    lengths = _pick_unique_lengths(
        {
            "top": (top_pool, top_count),
            "bottom": (bottom_pool, bottom_count),
            "left": (left_pool, left_count),
            "right": (right_pool, right_count),
        }
    )
    if lengths is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    palette = [c for c in range(1, 10)]
    random.shuffle(palette)
    body_colors = palette[:total]
    marker_colors = palette[total:]
    if not marker_colors:
        marker_colors = [1, 2, 3]

    top_cols = sorted(random.sample(list(range(c_lo, c_hi + 1)), top_count))
    bottom_cols = sorted(random.sample([c for c in range(c_lo, c_hi + 1) if c not in top_cols], bottom_count))
    left_rows = sorted(random.sample(list(range(r_lo, r_hi + 1)), left_count))
    right_rows = sorted(random.sample([r for r in range(r_lo, r_hi + 1) if r not in left_rows], right_count))

    placements = []
    color_idx = 0

    for col, length in zip(top_cols, lengths["top"]):
        placements.append((0, col, 1, 0, length, body_colors[color_idx]))
        color_idx += 1
    for col, length in zip(bottom_cols, lengths["bottom"]):
        placements.append((h - 1, col, -1, 0, length, body_colors[color_idx]))
        color_idx += 1
    for row, length in zip(left_rows, lengths["left"]):
        placements.append((row, 0, 0, 1, length, body_colors[color_idx]))
        color_idx += 1
    for row, length in zip(right_rows, lengths["right"]):
        placements.append((row, w - 1, 0, -1, length, body_colors[color_idx]))
        color_idx += 1

    random.shuffle(marker_colors)
    for idx, (r, c, dr, dc, length, body) in enumerate(placements):
        marker = marker_colors[idx % len(marker_colors)]
        if marker == body:
            marker = marker_colors[(idx + 1) % len(marker_colors)]
            if marker == body:
                marker = next(color for color in range(1, 10) if color != body)

        for step in range(length):
            rr = r + dr * step
            cc = c + dc * step
            if not (0 <= rr < h and 0 <= cc < w):
                return None
            value = marker if step == length - 1 else body
            if grid[rr, cc] != 0:
                return None
            grid[rr, cc] = value

    if not _validate_generated(grid, total):
        return None
    return grid


def _pick_unique_lengths(spec: dict[str, Tuple[List[int], int]]) -> dict[str, List[int]] | None:
    all_values = []
    for pool, _count in spec.values():
        all_values.extend(pool)
    if len(set(all_values)) < sum(count for _pool, count in spec.values()):
        return None

    for _ in range(200):
        used = set()
        result: dict[str, List[int]] = {}
        order = list(spec.keys())
        random.shuffle(order)
        ok = True
        for name in order:
            pool, count = spec[name]
            choices = [v for v in pool if v not in used]
            if len(choices) < count:
                ok = False
                break
            picked = random.sample(choices, count)
            used.update(picked)
            result[name] = picked
        if ok:
            for name in spec:
                result[name].sort()
            return result
    return None


def _validate_generated(grid: np.ndarray, expected_paths: int) -> bool:
    try:
        records = _extract_paths(grid)
        if len(records) != expected_paths:
            return False
        lengths = [length for length, _marker in records]
        if len(lengths) != len(set(lengths)):
            return False
        out = main(grid)
        if out.shape != (1, expected_paths):
            return False
    except AssertionError:
        return False
    return True
