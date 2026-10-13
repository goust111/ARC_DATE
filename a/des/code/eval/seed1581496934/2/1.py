from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9

LEGEND_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)

GRID_HEIGHT_RANGE = (11, 24)
GRID_WIDTH_RANGE = (12, 28)
LEGEND_LEN_RANGE = (3, 7)
RECT_OUTER_HEIGHT_RANGE = (4, 9)
RECT_OUTER_WIDTH_RANGE = (4, 12)
RECT_PLACEMENT_TRIES = 150
GEN_RETRIES = 300
OBJECT_TOP_MARGIN = 3
OBJECT_PAD = 1


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_legend_and_pattern(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    assert h >= 2, "grid must have legend row and pattern row"

    best: tuple[int, int] | None = None
    c = 0
    while c < w:
        if int(grid[0, c]) == BLACK:
            c += 1
            continue
        start = c
        while c < w and int(grid[0, c]) != BLACK:
            c += 1
        end = c
        legend = [int(v) for v in grid[0, start:end]]
        pattern = np.asarray(grid[1, start:end], dtype=int)
        if (
            len(legend) >= 3
            and len(set(legend)) == len(legend)
            and set(int(v) for v in pattern).issubset({BLACK, GREY})
            and GREY in pattern
        ):
            if best is None or (end - start) > (best[1] - best[0]):
                best = (start, end)

    assert best is not None, "failed to locate legend bar"
    start, end = best
    return [int(v) for v in grid[0, start:end]], np.asarray(grid[1, start:end], dtype=int)


def _rectangle_border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0 + 1, r1):
        border.add((r, c0))
        border.add((r, c1))
    return border


def _find_maroon_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components4(grid == MAROON):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = _rectangle_border_cells(r0, c0, r1, c1)
        if len(border) != len(comp):
            continue
        if set(comp) != border:
            continue
        frames.append((r0, c0, r1, c1))
    frames.sort()
    return frames


def _filled_interior(pattern: np.ndarray, seed_color: int, inner_h: int, inner_w: int, shift: int) -> np.ndarray:
    shifted = np.roll(pattern, shift)
    row_pattern = shifted[np.arange(inner_w) % len(shifted)]
    fill = np.zeros((inner_h, inner_w), dtype=int)
    fill[:, row_pattern == GREY] = int(seed_color)
    return fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the seed color's position in the legend as a phase index for the binary pattern row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected 2D grid"

    legend_colors, pattern = _find_legend_and_pattern(g)
    color_to_index = {color: idx for idx, color in enumerate(legend_colors)}
    frames = _find_maroon_frames(g)
    assert frames, "expected at least one maroon rectangular frame"

    out = g.copy()
    for r0, c0, r1, c1 in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_points = np.argwhere(inner != BLACK)
        assert len(seed_points) == 1, "each frame must contain exactly one seed"
        sr, sc = map(int, seed_points[0])
        seed_color = int(inner[sr, sc])
        assert seed_color in color_to_index, "seed color must appear in the legend"

        fill = _filled_interior(
            pattern=pattern,
            seed_color=seed_color,
            inner_h=inner.shape[0],
            inner_w=inner.shape[1],
            shift=color_to_index[seed_color],
        )
        out[r0 + 1 : r1, c0 + 1 : c1] = fill

    return out


def _has_full_period(pattern: np.ndarray) -> bool:
    n = len(pattern)
    for shift in range(1, n):
        if np.array_equal(pattern, np.roll(pattern, shift)):
            return False
    return True


def _random_pattern(length: int, rng: random.Random) -> np.ndarray:
    for _ in range(100):
        pattern = np.array(
            [GREY if rng.random() < 0.5 else BLACK for _ in range(length)],
            dtype=int,
        )
        if (pattern == GREY).any() and (pattern == BLACK).any() and _has_full_period(pattern):
            return pattern
    raise RuntimeError("failed to sample a non-degenerate pattern")


def _target_rect_count(height: int, width: int, rng: random.Random) -> int:
    area = height * width
    if area < 180:
        choices = (1, 2)
    elif area < 280:
        choices = (2, 3)
    else:
        choices = (2, 3, 4)
    return rng.choice(choices)


def _pick_seed_colors(legend_colors: list[int], count: int, rng: random.Random) -> list[int]:
    colors = legend_colors[:]
    rng.shuffle(colors)
    picked: list[int] = []
    for i in range(count):
        if i < len(colors):
            picked.append(colors[i])
        else:
            picked.append(rng.choice(legend_colors))
    if count > 1 and len(set(picked)) == 1 and len(legend_colors) > 1:
        picked[-1] = legend_colors[(legend_colors.index(picked[0]) + 1) % len(legend_colors)]
    return picked


def _place_frame(
    grid: np.ndarray,
    occupied: np.ndarray,
    seed_color: int,
    rng: random.Random,
) -> tuple[int, int, int, int, int] | None:
    h, w = grid.shape
    max_outer_h = min(RECT_OUTER_HEIGHT_RANGE[1], h - OBJECT_TOP_MARGIN)
    max_outer_w = min(RECT_OUTER_WIDTH_RANGE[1], w)
    if max_outer_h < RECT_OUTER_HEIGHT_RANGE[0] or max_outer_w < RECT_OUTER_WIDTH_RANGE[0]:
        return None

    for _ in range(RECT_PLACEMENT_TRIES):
        outer_h = rng.randint(RECT_OUTER_HEIGHT_RANGE[0], max_outer_h)
        outer_w = rng.randint(RECT_OUTER_WIDTH_RANGE[0], max_outer_w)
        r0 = rng.randint(OBJECT_TOP_MARGIN, h - outer_h)
        c0 = rng.randint(0, w - outer_w)

        rr0 = max(0, r0 - OBJECT_PAD)
        cc0 = max(0, c0 - OBJECT_PAD)
        rr1 = min(h, r0 + outer_h + OBJECT_PAD)
        cc1 = min(w, c0 + outer_w + OBJECT_PAD)
        if occupied[rr0:rr1, cc0:cc1].any():
            continue

        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        grid[r0, c0 : c1 + 1] = MAROON
        grid[r1, c0 : c1 + 1] = MAROON
        grid[r0 : r1 + 1, c0] = MAROON
        grid[r0 : r1 + 1, c1] = MAROON

        seed_r = rng.randint(r0 + 1, r1 - 1)
        seed_c = rng.randint(c0 + 1, c1 - 1)
        grid[seed_r, seed_c] = seed_color
        occupied[rr0:rr1, cc0:cc1] = True
        return r0, c0, r1, c1, seed_color

    return None


def _expected_output(
    grid: np.ndarray,
    legend_colors: list[int],
    pattern: np.ndarray,
    records: list[tuple[int, int, int, int, int]],
) -> np.ndarray:
    color_to_index = {color: idx for idx, color in enumerate(legend_colors)}
    out = grid.copy()
    for r0, c0, r1, c1, seed_color in records:
        fill = _filled_interior(
            pattern=pattern,
            seed_color=seed_color,
            inner_h=r1 - r0 - 1,
            inner_w=c1 - c0 - 1,
            shift=color_to_index[seed_color],
        )
        out[r0 + 1 : r1, c0 + 1 : c1] = fill
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_HEIGHT_RANGE)
    width = rng.randint(*GRID_WIDTH_RANGE)
    legend_len = rng.randint(LEGEND_LEN_RANGE[0], min(LEGEND_LEN_RANGE[1], width))
    legend_c0 = rng.randint(0, width - legend_len)

    legend_colors = rng.sample(list(LEGEND_COLOR_POOL), legend_len)
    pattern = _random_pattern(legend_len, rng)

    grid = np.zeros((height, width), dtype=int)
    grid[0, legend_c0 : legend_c0 + legend_len] = legend_colors
    grid[1, legend_c0 : legend_c0 + legend_len] = pattern

    occupied = np.zeros_like(grid, dtype=bool)
    occupied[:OBJECT_TOP_MARGIN, :] = True

    rect_count = _target_rect_count(height, width, rng)
    seed_colors = _pick_seed_colors(legend_colors, rect_count, rng)

    records: list[tuple[int, int, int, int, int]] = []
    for seed_color in seed_colors:
        placed = _place_frame(grid, occupied, seed_color, rng)
        if placed is None:
            return None
        records.append(placed)

    detected_legend, detected_pattern = _find_legend_and_pattern(grid)
    if detected_legend != legend_colors or not np.array_equal(detected_pattern, pattern):
        return None
    if len(_find_maroon_frames(grid)) != len(records):
        return None

    expected = _expected_output(grid, legend_colors, pattern, records)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
