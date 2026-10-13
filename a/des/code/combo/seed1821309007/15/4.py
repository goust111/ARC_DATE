from __future__ import annotations

import random

import numpy as np


AZURE = 8
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _color_counts(grid: np.ndarray) -> dict[int, int]:
    vals, cnts = np.unique(grid, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnts)}


def _get_bg(grid: np.ndarray) -> int:
    counts = _color_counts(grid)
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _find_gate_pairs(grid: np.ndarray, bg: int, marker: int) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    pairs: list[tuple[int, int, int, int]] = []
    for r in range(h):
        non_bg_cols = [c for c in range(w) if int(grid[r, c]) != bg]
        if len(non_bg_cols) < 2:
            continue
        for i, c0 in enumerate(non_bg_cols):
            color = int(grid[r, c0])
            if color == marker:
                continue
            for c1 in non_bg_cols[i + 1 :]:
                if int(grid[r, c1]) != color or c1 - c0 < 2:
                    continue
                interior = grid[r, c0 + 1 : c1]
                if interior.size == 0:
                    continue
                if not np.all((interior == bg) | (interior == marker)):
                    continue
                outside_left = c0 == 0 or int(grid[r, c0 - 1]) == bg
                outside_right = c1 == w - 1 or int(grid[r, c1 + 1]) == bg
                if outside_left and outside_right:
                    pairs.append((r, c0, c1, color))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Interpret each same-color pixel pair as gate posts. Count grey markers between
    the posts, then draw an upside-down U frame whose height equals that count.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    pairs = _find_gate_pairs(g, bg, GREY)
    assert pairs, "expected at least one gate-post pair"

    for r, c0, c1, color in pairs:
        interior = g[r, c0 + 1 : c1]
        n = int(np.sum(interior == GREY))
        top_r = r - n
        assert top_r >= 0, "gate height must fit above the posts"

        out[r, c0 + 1 : c1] = bg
        out[top_r, c0 : c1 + 1] = color
        out[top_r : r + 1, c0] = color
        out[top_r : r + 1, c1] = color

    return out


def _segment_free(row_used: np.ndarray, c0: int, c1: int, margin: int = 1) -> bool:
    left = max(0, c0 - margin)
    right = min(len(row_used), c1 + margin + 1)
    return not bool(row_used[left:right].any())


def _paint_gate(
    grid: np.ndarray,
    row_used: np.ndarray,
    r: int,
    c0: int,
    c1: int,
    color: int,
    marker_cols: list[int],
) -> None:
    grid[r, c0] = color
    grid[r, c1] = color
    for c in marker_cols:
        grid[r, c] = GREY
    row_used[c0 : c1 + 1] = True


def _sample_marker_cols(c0: int, c1: int, count: int) -> list[int]:
    interior = list(range(c0 + 1, c1))
    if count == 0:
        return []
    return sorted(random.sample(interior, count))


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(10, 22)
    w = random.randint(10, 24)
    grid = np.full((h, w), AZURE, dtype=int)

    max_gates = min(5, h - 2)
    gate_count = random.randint(1, max_gates)
    rows = random.sample(range(1, h), gate_count)
    rows.sort()
    random.shuffle(rows)

    row_used = np.zeros((h, w), dtype=bool)
    used_colors: list[int] = []
    tall = False
    zero_height = False

    for idx, r in enumerate(rows):
        remaining = gate_count - idx - 1
        candidate_ns = list(range(0, min(5, r) + 1))
        random.shuffle(candidate_ns)
        if not tall:
            candidate_ns.sort(key=lambda n: (n < 2, random.random()))
        elif not zero_height:
            candidate_ns.sort(key=lambda n: (n != 0, random.random()))

        placed = False
        for n in candidate_ns:
            min_gap = max(2, n + 1)
            gap = random.randint(min_gap, min(min_gap + 6, w - 3))
            c0_choices = list(range(1, w - gap - 1))
            random.shuffle(c0_choices)
            color_pool = PALETTE.copy()
            random.shuffle(color_pool)

            for c0 in c0_choices:
                c1 = c0 + gap
                if c1 >= w - 1:
                    continue
                if not _segment_free(row_used[r], c0, c1):
                    continue
                if n > 0 and row_used[r - n : r + 1, c0].any():
                    continue
                if n > 0 and row_used[r - n : r + 1, c1].any():
                    continue
                if row_used[max(0, r - n), c0 : c1 + 1].any():
                    continue

                marker_cols = _sample_marker_cols(c0, c1, n)
                for color in color_pool:
                    if color == GREY:
                        continue
                    if used_colors and random.random() < 0.5:
                        color = random.choice(used_colors)
                    _paint_gate(grid, row_used, r, c0, c1, color, marker_cols)
                    used_colors.append(color)
                    tall = tall or n >= 2
                    zero_height = zero_height or n == 0
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break

        if not placed:
            return None

    if not (tall and zero_height) and gate_count >= 2:
        return None

    pairs = _find_gate_pairs(grid, AZURE, GREY)
    if len(pairs) != gate_count:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
