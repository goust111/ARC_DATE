from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


Square = tuple[int, int, int]  # (top_row, left_col, side)
Source = tuple[int, int, int]  # (row, col, color)


BLACK = 0
NON_BLACK_COLORS = tuple(range(1, 10))
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

GEN_HEIGHT_RANGE = (14, 20)
GEN_WIDTH_RANGE = (14, 20)
GEN_SOURCE_COUNT_CHOICES = (2, 3)
GEN_MARGIN_CHOICES = (2, 3, 4)
GEN_MAX_ATTEMPTS = 400
GEN_POS_ATTEMPTS = 60
GEN_MIN_COVERAGE = 0.18
GEN_MAX_COVERAGE = 0.72
GEN_MAX_SELECTED = 18


def _overlap(a: Square, b: Square) -> bool:
    ar, ac, asz = a
    br, bc, bsz = b
    return ar < br + bsz and br < ar + asz and ac < bc + bsz and bc < ac + asz


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _source_positions(grid: np.ndarray) -> list[Source]:
    coords = np.argwhere(grid != BLACK)
    sources = [(int(r), int(c), int(grid[r, c])) for r, c in coords]
    sources.sort()
    colors = [color for _, _, color in sources]
    assert 2 <= len(sources) <= 3, "expected 2-3 isolated source pixels"
    assert len(colors) == len(set(colors)), "source colors must be distinct"

    for r, c, _color in sources:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if _in_bounds(grid.shape, nr, nc):
                    assert int(grid[nr, nc]) == BLACK, "source pixels must be isolated"
    return sources


def _draw_rays(grid: np.ndarray, sources: list[Source]) -> tuple[np.ndarray, int]:
    canvas = np.array(grid, dtype=int, copy=True)
    collision_count = 0
    for r, c, color in sources:
        for dr, dc in DIAGONALS:
            rr, cc = r + dr, c + dc
            while _in_bounds(canvas.shape, rr, cc):
                if int(canvas[rr, cc]) == BLACK:
                    canvas[rr, cc] = color
                else:
                    collision_count += 1
                rr += dr
                cc += dc
    return canvas, collision_count


def _black_square_candidates(ray_grid: np.ndarray) -> list[Square]:
    h, w = ray_grid.shape
    candidates: list[Square] = []
    for side in range(min(h, w), 1, -1):
        for r in range(h - side + 1):
            for c in range(w - side + 1):
                if np.all(ray_grid[r : r + side, c : c + side] == BLACK):
                    candidates.append((r, c, side))
    return candidates


def _greedy_select_squares(candidates: list[Square]) -> list[Square]:
    selected: list[Square] = []
    for square in candidates:
        if any(_overlap(square, other) for other in selected):
            continue
        selected.append(square)
    return selected


def _same_size_overlap_exists(candidates: list[Square]) -> bool:
    by_side: dict[int, list[Square]] = defaultdict(list)
    for square in candidates:
        by_side[square[2]].append(square)
    for squares in by_side.values():
        for i, square in enumerate(squares):
            for other in squares[i + 1 :]:
                if _overlap(square, other):
                    return True
    return False


def _adjacent_ray_color_counts(square: Square, ray_grid: np.ndarray) -> Counter[int]:
    h, w = ray_grid.shape
    r0, c0, side = square
    counts: Counter[int] = Counter()

    for c in range(c0, c0 + side):
        for nr, nc in ((r0 - 1, c), (r0 + side, c)):
            if _in_bounds((h, w), nr, nc):
                color = int(ray_grid[nr, nc])
                if color != BLACK:
                    counts[color] += 1

    for r in range(r0, r0 + side):
        for nr, nc in ((r, c0 - 1), (r, c0 + side)):
            if _in_bounds((h, w), nr, nc):
                color = int(ray_grid[nr, nc])
                if color != BLACK:
                    counts[color] += 1

    return counts


def _choose_fill_color(square: Square, ray_grid: np.ndarray, source_rank: dict[int, tuple[int, int]]) -> tuple[int, bool]:
    counts = _adjacent_ray_color_counts(square, ray_grid)
    assert counts, "selected square must touch at least one outward-adjacent ray pixel"
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    tied = len(winners) > 1
    winners.sort(key=lambda color: source_rank[color])
    return winners[0], tied


def _analyze(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    sources = _source_positions(g)
    source_rank = {color: (r, c) for r, c, color in sources}
    ray_grid, collision_count = _draw_rays(g, sources)
    candidates = _black_square_candidates(ray_grid)
    selected = _greedy_select_squares(candidates)

    fill_colors: list[int] = []
    tie_flags: list[bool] = []
    for square in selected:
        color, tied = _choose_fill_color(square, ray_grid, source_rank)
        fill_colors.append(color)
        tie_flags.append(tied)

    return {
        "sources": sources,
        "source_rank": source_rank,
        "ray_grid": ray_grid,
        "candidates": candidates,
        "selected": selected,
        "fill_colors": fill_colors,
        "tie_flags": tie_flags,
        "collision_count": collision_count,
        "same_size_overlap": _same_size_overlap_exists(candidates),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Draw source-colored diagonal rays, greedily fill the largest black square gaps, and color each square by its neighboring ray majority.

    Input structure:
    - The grid has a black background with 2-3 isolated non-black source pixels of distinct colors.
    - Each source pixel defines four diagonal ray directions through its own location.
    - No other foreground objects are present in the input.

    Rule:
    - Sort the source pixels by row, then column, and draw their four diagonal rays in that order.
    - Earlier rays keep priority: if a later ray reaches an already colored cell, that cell keeps its current color while the ray continues farther in the same direction.
    - After all rays are drawn, search all all-black axis-aligned squares of side at least 2 and greedily keep a largest-first, non-overlapping set; within the same side length, earlier top-left positions win.
    - Fill each kept square with the most frequent source color among the outward 4-neighbor ray pixels touching its boundary; break color ties by the top-left order of the corresponding source pixels.

    Key insight:
    - The same diagonal ray layout simultaneously carves the black gaps that determine square size and provides the surrounding color evidence that determines each chosen square's fill.
    """
    info = _analyze(np.asarray(grid, dtype=int))
    out = np.array(info["ray_grid"], copy=True)
    for square, color in zip(info["selected"], info["fill_colors"], strict=False):
        r0, c0, side = square
        out[r0 : r0 + side, c0 : c0 + side] = int(color)
    return out


def _sample_positions(height: int, width: int, n_sources: int) -> list[tuple[int, int]] | None:
    for _ in range(GEN_POS_ATTEMPTS):
        margin = random.choice(GEN_MARGIN_CHOICES)
        if height <= 2 * margin + 2 or width <= 2 * margin + 2:
            continue

        mid_r = random.randint(margin + 1, height - margin - 2)
        mid_c = random.randint(margin + 1, width - margin - 2)
        anchors = [
            (margin, margin),
            (margin, width - 1 - margin),
            (height - 1 - margin, margin),
            (height - 1 - margin, width - 1 - margin),
            (margin, mid_c),
            (height - 1 - margin, mid_c),
            (mid_r, margin),
            (mid_r, width - 1 - margin),
        ]

        random.shuffle(anchors)
        positions: list[tuple[int, int]] = []
        for r, c in anchors:
            if any(max(abs(r - rr), abs(c - cc)) <= 2 for rr, cc in positions):
                continue
            positions.append((r, c))
            if len(positions) == n_sources:
                return positions
    return None


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        info = _analyze(grid)
    except AssertionError:
        return False

    selected: list[Square] = info["selected"]  # type: ignore[assignment]
    fill_colors: list[int] = info["fill_colors"]  # type: ignore[assignment]
    tie_flags: list[bool] = info["tie_flags"]  # type: ignore[assignment]
    height, width = grid.shape
    total_area = height * width
    coverage = sum(side * side for _, _, side in selected) / total_area

    if len(selected) < 2:
        return False
    if len(selected) > GEN_MAX_SELECTED:
        return False
    if max(side for _, _, side in selected) < 3:
        return False
    if not (GEN_MIN_COVERAGE <= coverage <= GEN_MAX_COVERAGE):
        return False
    if len(set(fill_colors)) < 2:
        return False
    if not any(tie_flags):
        return False
    if not any(not tied for tied in tie_flags):
        return False
    if int(info["collision_count"]) <= 0:
        return False
    if not bool(info["same_size_overlap"]):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    n_sources = random.choice(GEN_SOURCE_COUNT_CHOICES)
    positions = _sample_positions(height, width, n_sources)
    if positions is None:
        return None

    colors = random.sample(NON_BLACK_COLORS, n_sources)
    grid = np.zeros((height, width), dtype=int)
    for (r, c), color in zip(positions, colors, strict=False):
        grid[r, c] = color

    if _semantic_ok(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
