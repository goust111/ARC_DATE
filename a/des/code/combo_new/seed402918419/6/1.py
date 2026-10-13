import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
PATTERN_KINDS = ("hstripe", "vstripe", "checker", "diag")
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _tile_pattern(pattern: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    ph, pw = pattern.shape
    rows = np.arange(shape[0]) % ph
    cols = np.arange(shape[1]) % pw
    return pattern[rows[:, None], cols[None, :]]


def _extract_legend_map(grid: np.ndarray) -> dict[int, np.ndarray]:
    grey_positions = np.argwhere(grid == GREY)
    assert grey_positions.size > 0, "expected at least one grey fence"
    fence_top = int(grey_positions[:, 0].min())

    top_band = grid[:fence_top, :]
    comps = _components((top_band != BG) & (top_band != GREY))

    mapping: dict[int, np.ndarray] = {}
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        patch = top_band[r0 : r1 + 1, c0 : c1 + 1]
        h, w = patch.shape
        if w <= h:
            continue
        square = patch[:, :h]
        pattern = patch[:, h:]
        if pattern.shape[1] == 0:
            continue
        if not np.all(square == square[0, 0]):
            continue
        seed_color = int(square[0, 0])
        pattern_colors = set(map(int, np.unique(pattern)))
        if BG in pattern_colors or GREY in pattern_colors:
            continue
        if seed_color not in pattern_colors or len(pattern_colors) != 2:
            continue
        assert seed_color not in mapping, "legend colors must be unique"
        mapping[seed_color] = pattern.copy()

    assert mapping, "failed to parse any legend entries"
    return mapping


def _find_fences(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    fences: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border_ok = (
            np.all(patch[0, :] == GREY)
            and np.all(patch[-1, :] == GREY)
            and np.all(patch[:, 0] == GREY)
            and np.all(patch[:, -1] == GREY)
        )
        if not border_ok:
            continue
        if int(np.count_nonzero(patch == GREY)) != 2 * h + 2 * w - 4:
            continue
        fences.append((r0, c0, r1, c1))
    assert fences, "failed to detect any fenced rectangles"
    return fences


def _make_pattern(kind: str, height: int, width: int, primary: int, secondary: int) -> np.ndarray:
    rr, cc = np.indices((height, width))
    if kind == "hstripe":
        mask = rr % 2 == 0
    elif kind == "vstripe":
        mask = cc % 2 == 0
    elif kind == "checker":
        mask = (rr + cc) % 2 == 0
    elif kind == "diag":
        mask = (rr + cc) % 3 == 0
    else:
        raise ValueError(f"unknown pattern kind: {kind}")
    pattern = np.where(mask, primary, secondary)
    assert np.any(pattern == primary) and np.any(pattern == secondary)
    return pattern


def _crop_with_padding(grid: np.ndarray) -> np.ndarray:
    pts = np.argwhere(grid != BG)
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0 = max(0, int(r0) - 1)
    c0 = max(0, int(c0) - 1)
    r1 = min(grid.shape[0] - 1, int(r1) + 1)
    c1 = min(grid.shape[1] - 1, int(c1) + 1)
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey fenced region by tiling the legend pattern selected by its seed color.

    Input structure:
    - The background is black.
    - A legend row near the top contains several entries; each entry is a solid colored square immediately followed by a rectangular two-color pattern block.
    - Lower in the grid are one-pixel-thick grey rectangular fences, each enclosing exactly one colored seed pixel on black background.

    Rule:
    - Build a mapping from each legend square color to the adjacent pattern block.
    - For every grey fence, read the unique seed color inside the fence and find the matching legend entry.
    - Tile that pattern from the interior's top-left corner across the entire fenced interior, overwriting the seed pixel as part of the fill.
    - Leave the legend row, grey borders, and all outside background unchanged.

    Key insight:
    - The interior seed pixel is only a lookup key; the actual content comes from the top legend's adjacent pattern tile, which must then be repeated spatially.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    legend_map = _extract_legend_map(g)
    for r0, c0, r1, c1 in _find_fences(g):
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_positions = np.argwhere((interior != BG) & (interior != GREY))
        assert seed_positions.shape[0] == 1, "each fence must contain exactly one seed"
        sr, sc = map(int, seed_positions[0])
        seed_color = int(interior[sr, sc])
        assert seed_color in legend_map, "every seed color must have a legend entry"
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_pattern(legend_map[seed_color], interior.shape)

    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    canvas = np.zeros((30, 30), dtype=int)

    legend_count = rng.randint(2, 4)
    square_size = rng.choice([2, 3])
    seed_colors = rng.sample(COLOR_POOL, legend_count)
    accent_colors = rng.sample([c for c in COLOR_POOL if c not in seed_colors], legend_count)

    legend_specs: list[tuple[int, int, np.ndarray]] = []
    total_width = 0
    for idx in range(legend_count):
        pattern_width = rng.randint(2, 4)
        kind = rng.choice(PATTERN_KINDS)
        pattern = _make_pattern(kind, square_size, pattern_width, seed_colors[idx], accent_colors[idx])
        entry_width = square_size + pattern_width
        gap = 0 if idx == legend_count - 1 else rng.randint(1, 2)
        legend_specs.append((seed_colors[idx], gap, pattern))
        total_width += entry_width + gap
    if total_width > 28:
        return None

    top_row = rng.randint(1, 2)
    start_col = rng.randint(1, 29 - total_width)
    occupied = np.zeros_like(canvas, dtype=bool)
    col = start_col
    for seed_color, gap, pattern in legend_specs:
        h, w = pattern.shape
        canvas[top_row : top_row + h, col : col + h] = seed_color
        canvas[top_row : top_row + h, col + h : col + h + w] = pattern
        r0 = max(0, top_row - 1)
        r1 = min(30, top_row + h + 1)
        c0 = max(0, col - 1)
        c1 = min(30, col + h + w + 1)
        occupied[r0:r1, c0:c1] = True
        col += h + w + gap

    fence_count = rng.randint(2, 4)
    fence_top_min = top_row + square_size + 3
    if fence_top_min > 24:
        return None

    used_colors = seed_colors[:]
    rng.shuffle(used_colors)
    while len(used_colors) < fence_count:
        used_colors.append(rng.choice(seed_colors))

    for fence_idx in range(fence_count):
        interior_h = rng.randint(2, 6)
        interior_w = rng.randint(2, 7)
        outer_h = interior_h + 2
        outer_w = interior_w + 2
        placed = False
        for _ in range(200):
            if fence_top_min > 29 - outer_h:
                return None
            r0 = rng.randint(fence_top_min, 29 - outer_h)
            c0 = rng.randint(1, 29 - outer_w)
            pr0 = max(0, r0 - 1)
            pr1 = min(30, r0 + outer_h + 1)
            pc0 = max(0, c0 - 1)
            pc1 = min(30, c0 + outer_w + 1)
            if np.any(occupied[pr0:pr1, pc0:pc1]):
                continue

            canvas[r0, c0 : c0 + outer_w] = GREY
            canvas[r0 + outer_h - 1, c0 : c0 + outer_w] = GREY
            canvas[r0 : r0 + outer_h, c0] = GREY
            canvas[r0 : r0 + outer_h, c0 + outer_w - 1] = GREY

            seed_r = rng.randint(r0 + 1, r0 + outer_h - 2)
            seed_c = rng.randint(c0 + 1, c0 + outer_w - 2)
            canvas[seed_r, seed_c] = used_colors[fence_idx]
            occupied[pr0:pr1, pc0:pc1] = True
            placed = True
            break
        if not placed:
            return None

    grid = _crop_with_padding(canvas)
    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    if grid.ndim != 2 or max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
