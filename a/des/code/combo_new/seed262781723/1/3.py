from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GEN_SEPARATOR = 5
GEN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _find_separator_color(grid: np.ndarray, bg: int = BG) -> int:
    colors: set[int] = set()
    for row in grid:
        vals = np.unique(row)
        if vals.size == 1 and int(vals[0]) != int(bg):
            colors.add(int(vals[0]))
    for col in grid.T:
        vals = np.unique(col)
        if vals.size == 1 and int(vals[0]) != int(bg):
            colors.add(int(vals[0]))
    assert len(colors) == 1, "expected a unique separator color"
    return next(iter(colors))


def _line_indices(grid: np.ndarray, color: int) -> tuple[list[int], list[int]]:
    row_lines = [int(r) for r in range(grid.shape[0]) if bool(np.all(grid[r, :] == int(color)))]
    col_lines = [int(c) for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == int(color)))]
    return row_lines, col_lines


def _spans_from_lines(length: int, lines: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for line in sorted(set(int(x) for x in lines)):
        if start < int(line):
            spans.append((int(start), int(line)))
        start = int(line) + 1
    if start < int(length):
        spans.append((int(start), int(length)))
    return spans


def _winner_color(region: np.ndarray, bg: int, sep: int) -> int | None:
    mask = (region != int(bg)) & (region != int(sep))
    if not bool(mask.any()):
        return None

    stats: dict[int, tuple[int, tuple[int, int]]] = {}
    for r, c in np.argwhere(mask):
        color = int(region[int(r), int(c)])
        count, first = stats.get(color, (0, (int(r), int(c))))
        first = min(first, (int(r), int(c)))
        stats[color] = (int(count) + 1, first)

    winner = min(
        stats.items(),
        key=lambda item: (-item[1][0], item[1][1][0], item[1][1][1]),
    )[0]
    return int(winner)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each separator-delimited region with its local majority color while leaving vote-cross rows and columns black.

    Input structure:
    - The background is black.
    - One or more solid full rows and/or full columns in a single separator color divide the grid into rectangular regions.
    - Inside each region, a few non-background, non-separator colored pixels act as votes; each region uses two or three vote colors.

    Rule:
    - Detect the separator color from the full rows/columns and recover every rectangular region between those lines.
    - In each region, count the vote pixels by color; the color with the largest count is the fill color, and count ties are broken by whichever tied color appears first from top to bottom.
    - Fill only those black background cells whose row and column are both unused by every original vote pixel in that region.
    - Keep the separator lines and all original vote pixels unchanged.

    Key insight:
    - The same sparse vote pixels play two opposite roles: they elect the region's fill color, but their own rows and columns also carve permanent black cross-shaped holes through that fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = BG
    sep = _find_separator_color(g, bg)
    row_lines, col_lines = _line_indices(g, sep)
    row_spans = _spans_from_lines(int(g.shape[0]), row_lines)
    col_spans = _spans_from_lines(int(g.shape[1]), col_lines)

    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = g[int(r0) : int(r1), int(c0) : int(c1)]
            winner = _winner_color(region, bg, sep)
            if winner is None:
                continue

            seed_mask = (region != int(bg)) & (region != int(sep))
            blocked_rows = np.any(seed_mask, axis=1)
            blocked_cols = np.any(seed_mask, axis=0)
            fill_mask = (region == int(bg)) & (~blocked_rows[:, None]) & (~blocked_cols[None, :])
            out[int(r0) : int(r1), int(c0) : int(c1)][fill_mask] = int(winner)

    return out


def _counts_for_branch(n_colors: int, branch: str) -> list[int]:
    if branch == "tie":
        top_count = random.choice([2, 3])
        counts = [top_count, top_count]
        if int(n_colors) == 3:
            counts.append(random.randint(1, top_count - 1))
        return counts

    if int(n_colors) == 2:
        winner_count = random.choice([3, 4])
        counts = [winner_count, random.randint(1, winner_count - 1)]
        return counts

    winner_count = random.choice([3, 4])
    counts = [winner_count, random.randint(1, 2), random.randint(1, 2)]
    if max(counts[1:]) >= winner_count:
        counts[1] = 1
        counts[2] = 1
    return counts


def _build_region(height: int, width: int, branch: str) -> np.ndarray | None:
    for _ in range(120):
        region = np.full((int(height), int(width)), BG, dtype=int)
        n_colors = random.choice([2, 3])
        colors = random.sample(GEN_COLORS, n_colors)
        counts = _counts_for_branch(n_colors, branch)
        total = sum(counts)
        if total > 9:
            continue

        rows = sorted(random.sample(range(int(height)), 3))
        cols = sorted(random.sample(range(int(width)), 3))
        positions = [(int(r), int(c)) for r in rows for c in cols]

        if branch == "tie":
            top_colors = colors[:2]
            chosen_winner = random.choice(top_colors)
            other_top = top_colors[1 - top_colors.index(chosen_winner)]
            anchors = {
                int(chosen_winner): (int(rows[0]), int(random.choice(cols))),
                int(other_top): (int(rows[1]), int(random.choice(cols))),
            }
            assignment: dict[tuple[int, int], int] = {anchors[int(chosen_winner)]: int(chosen_winner)}
            if anchors[int(other_top)] in assignment:
                continue
            assignment[anchors[int(other_top)]] = int(other_top)

            available = [p for p in positions if p not in assignment]
            random.shuffle(available)

            need: list[tuple[int, int]] = [
                (int(chosen_winner), int(counts[colors.index(chosen_winner)]) - 1),
                (int(other_top), int(counts[colors.index(other_top)]) - 1),
            ]
            if int(n_colors) == 3:
                need.append((int(colors[2]), int(counts[2])))

            ok = True
            for color, amount in need:
                if amount <= 0:
                    continue
                if int(color) == int(other_top):
                    pool = [p for p in available if p[0] >= int(rows[1])]
                else:
                    pool = available[:]
                if len(pool) < int(amount):
                    ok = False
                    break
                chosen = random.sample(pool, int(amount))
                for pos in chosen:
                    assignment[pos] = int(color)
                    available.remove(pos)
            if not ok:
                continue

            intended_winner = int(chosen_winner)
        else:
            winner_color = int(colors[0])
            assignment = {}
            chosen_positions = random.sample(positions, int(total))
            random.shuffle(chosen_positions)
            idx = 0
            for color, amount in zip(colors, counts):
                for _ in range(int(amount)):
                    assignment[chosen_positions[idx]] = int(color)
                    idx += 1
            intended_winner = winner_color

        for (r, c), color in assignment.items():
            region[int(r), int(c)] = int(color)

        seed_mask = region != int(BG)
        distinct_rows = int(np.sum(np.any(seed_mask, axis=1)))
        distinct_cols = int(np.sum(np.any(seed_mask, axis=0)))
        fill_mask = (region == int(BG)) & (~np.any(seed_mask, axis=1)[:, None]) & (~np.any(seed_mask, axis=0)[None, :])
        if distinct_rows < 2 or distinct_cols < 2 or not bool(fill_mask.any()):
            continue

        derived = _winner_color(region, BG, GEN_SEPARATOR)
        if derived != int(intended_winner):
            continue

        if int(len(np.unique(region[seed_mask]))) != int(n_colors):
            continue
        return region
    return None


def _build_puzzle() -> np.ndarray | None:
    n_region_rows = random.choice([2, 3])
    n_region_cols = random.choice([2, 3])
    row_sizes = [random.randint(5, 8) for _ in range(int(n_region_rows))]
    col_sizes = [random.randint(5, 8) for _ in range(int(n_region_cols))]

    height = int(sum(row_sizes) + n_region_rows - 1)
    width = int(sum(col_sizes) + n_region_cols - 1)
    if max(height, width) > 30:
        return None

    grid = np.full((int(height), int(width)), BG, dtype=int)

    r = 0
    for i, size in enumerate(row_sizes[:-1]):
        r += int(size)
        grid[int(r), :] = int(GEN_SEPARATOR)
        r += 1

    c = 0
    for i, size in enumerate(col_sizes[:-1]):
        c += int(size)
        grid[:, int(c)] = int(GEN_SEPARATOR)
        c += 1

    branches = [random.choice(["tie", "majority"]) for _ in range(int(n_region_rows * n_region_cols))]
    if len(branches) >= 2 and len(set(branches)) == 1:
        branches[random.randrange(len(branches))] = "tie" if branches[0] == "majority" else "majority"

    row_starts = []
    r0 = 0
    for size in row_sizes:
        row_starts.append(r0)
        r0 += int(size) + 1

    col_starts = []
    c0 = 0
    for size in col_sizes:
        col_starts.append(c0)
        c0 += int(size) + 1

    branch_idx = 0
    for rr, rh in zip(row_starts, row_sizes):
        for cc, cw in zip(col_starts, col_sizes):
            region = _build_region(int(rh), int(cw), branches[branch_idx])
            if region is None:
                return None
            grid[int(rr) : int(rr + rh), int(cc) : int(cc + cw)] = region
            branch_idx += 1

    return grid


def _validate_generated(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    sep = _find_separator_color(grid, BG)
    row_lines, col_lines = _line_indices(grid, sep)
    assert row_lines and col_lines

    row_spans = _spans_from_lines(int(grid.shape[0]), row_lines)
    col_spans = _spans_from_lines(int(grid.shape[1]), col_lines)
    assert 2 <= len(row_spans) <= 3
    assert 2 <= len(col_spans) <= 3

    saw_tie = False
    saw_majority = False
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = grid[int(r0) : int(r1), int(c0) : int(c1)]
            seed_mask = region != int(BG)
            colors = Counter(int(v) for v in region[seed_mask].reshape(-1))
            assert 2 <= len(colors) <= 3
            counts = list(colors.values())
            if max(counts) == sorted(counts)[-2]:
                saw_tie = True
            else:
                saw_majority = True

            blocked_rows = np.any(seed_mask, axis=1)
            blocked_cols = np.any(seed_mask, axis=0)
            fill_mask = (region == int(BG)) & (~blocked_rows[:, None]) & (~blocked_cols[None, :])
            assert bool(fill_mask.any())

    assert saw_tie and saw_majority
    out = main(grid)
    assert out.shape == grid.shape
    assert bool(np.any(out != grid))


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_puzzle()
        if grid is None:
            continue
        try:
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
