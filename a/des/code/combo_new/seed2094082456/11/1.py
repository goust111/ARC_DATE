from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG_COLOR = 0
WALL_COLOR = 5  # grey
LEGEND_COLORS = (2, 1, 3, 4, 7)  # red, blue, green, yellow, orange
INTERIOR_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (16, 28)
GRID_W_RANGE = (18, 30)
REGION_COUNT_OPTIONS = (1, 2, 2, 3, 3, 4)
PATCH_ATTEMPTS = 100
PLACEMENT_ATTEMPTS = 150
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _background_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.flat))
    return counts.most_common(1)[0][0]


def _extract_legend(grid: np.ndarray, bg: int) -> list[int]:
    h, w = grid.shape
    candidates: list[list[int]] = []
    for col in (0, w - 1):
        run_start: int | None = None
        for r in range(h + 1):
            v = bg if r == h else int(grid[r, col])
            if r < h and v not in (bg, WALL_COLOR):
                if run_start is None:
                    run_start = r
                continue
            if run_start is not None:
                rows = list(range(run_start, r))
                if len(rows) == len(LEGEND_COLORS):
                    colors = [int(grid[rr, col]) for rr in rows]
                    if len(set(colors)) == len(LEGEND_COLORS):
                        others_ok = all(
                            int(grid[rr, col]) == bg
                            for rr in range(h)
                            if rr < run_start or rr >= r
                        )
                        if others_ok:
                            candidates.append(colors)
                run_start = None
    assert len(candidates) == 1, "expected a unique side legend"
    return candidates[0]


def _enclosed_regions(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    regions: list[list[tuple[int, int]]] = []
    for comp in _components(grid != WALL_COLOR):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue
        regions.append(comp)
    return regions


def _trapped_color_count(grid: np.ndarray, cells: list[tuple[int, int]], bg: int) -> int:
    colors = {
        int(grid[r, c])
        for r, c in cells
        if int(grid[r, c]) not in (bg, WALL_COLOR)
    }
    return len(colors)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each grey-enclosed region using the legend color indexed by its trapped color count.

    Input structure:
    - The background is a dominant plain color, typically black.
    - Several grey 4-connected fence loops enclose interior regions that contain scattered colored pixels and empty background cells.
    - One side edge contains a vertical 5-cell legend whose colors are ordered from top to bottom and serve as lookup entries for counts 1..5.

    Rule:
    - Read the side legend from top to bottom to obtain the ordered list of fill colors.
    - Treat grey cells as walls and find every 4-connected non-grey region that does not touch the grid boundary.
    - For each enclosed region, count how many distinct colors appear inside it after excluding the background and the grey wall color.
    - Fill the entire enclosed region, including its original dots and empty cells, with the legend's N-th color where N is that distinct-color count.

    Key insight:
    - The grey fences define the relevant closed regions, and the trapped color variety is not itself preserved; it is only used as a 1-based index into the visible edge legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    legend = _extract_legend(g, bg)

    out = g.copy()
    for region in _enclosed_regions(g):
        count = _trapped_color_count(g, region, bg)
        assert 1 <= count <= len(legend), "trapped color count must index the legend"
        fill = legend[count - 1]
        for r, c in region:
            out[r, c] = fill
    return out


def _draw_base_frame(inner_h: int, inner_w: int) -> np.ndarray:
    patch = np.full((inner_h + 4, inner_w + 4), BG_COLOR, dtype=int)
    top, left = 1, 1
    bottom, right = inner_h + 2, inner_w + 2
    patch[top, left:right + 1] = WALL_COLOR
    patch[bottom, left:right + 1] = WALL_COLOR
    patch[top:bottom + 1, left] = WALL_COLOR
    patch[top:bottom + 1, right] = WALL_COLOR
    return patch


def _add_notch(rng: random.Random, patch: np.ndarray, inner_h: int, inner_w: int) -> bool:
    bottom, right = inner_h + 2, inner_w + 2
    side = rng.choice(("top", "bottom", "left", "right"))
    if side in ("top", "bottom"):
        max_depth = min(2, inner_h - 2)
        max_span = min(3, inner_w - 2)
        if max_depth < 1 or max_span < 1:
            return False
        depth = rng.randint(1, max_depth)
        span = rng.randint(1, max_span)
        start = rng.randint(2, right - span)
        rows = slice(2, 2 + depth) if side == "top" else slice(bottom - depth, bottom)
        patch[rows, start:start + span] = WALL_COLOR
        return True
    max_depth = min(2, inner_w - 2)
    max_span = min(3, inner_h - 2)
    if max_depth < 1 or max_span < 1:
        return False
    depth = rng.randint(1, max_depth)
    span = rng.randint(1, max_span)
    start = rng.randint(2, bottom - span)
    cols = slice(2, 2 + depth) if side == "left" else slice(right - depth, right)
    patch[start:start + span, cols] = WALL_COLOR
    return True


def _make_region_patch(rng: random.Random, target_count: int) -> np.ndarray | None:
    for _ in range(PATCH_ATTEMPTS):
        inner_h = rng.randint(max(4, target_count + 1), 8)
        inner_w = rng.randint(4, 8)
        patch = _draw_base_frame(inner_h, inner_w)

        notch_attempts = rng.randint(1, 3)
        for _ in range(notch_attempts):
            _add_notch(rng, patch, inner_h, inner_w)

        wall_comps = _components(patch == WALL_COLOR)
        if len(wall_comps) != 1:
            continue

        regions = _enclosed_regions(patch)
        if len(regions) != 1:
            continue
        region = regions[0]
        if len(region) < target_count + 2:
            continue
        if not np.any(patch[2:-2, 2:-2] == WALL_COLOR):
            continue

        region_cells = region[:]
        rng.shuffle(region_cells)
        chosen_colors = list(rng.sample(INTERIOR_COLOR_POOL, target_count))
        for (r, c), color in zip(region_cells, chosen_colors):
            patch[r, c] = color

        spare = region_cells[target_count:]
        extra_count = rng.randint(0, min(4, len(spare)))
        for r, c in spare[:extra_count]:
            patch[r, c] = rng.choice(chosen_colors)

        if _trapped_color_count(patch, region, BG_COLOR) != target_count:
            continue
        return patch
    return None


def _choose_target_counts(rng: random.Random, n_regions: int) -> list[int]:
    if n_regions <= 1:
        return [rng.randint(1, 5)]
    if rng.random() < 0.7:
        counts = rng.sample(range(1, 6), n_regions)
    else:
        counts = [rng.randint(1, 5) for _ in range(n_regions)]
        if len(set(counts)) == 1:
            counts[0] = counts[0] % 5 + 1
    rng.shuffle(counts)
    return counts


def _place_patch(
    rng: random.Random,
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    legend_side: str,
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    row_min, row_max = 1, h - ph - 1
    if legend_side == "left":
        col_min, col_max = 2, w - pw - 2
    else:
        col_min, col_max = 1, w - pw - 3
    if row_max < row_min or col_max < col_min:
        return False

    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = rng.randint(row_min, row_max)
        c0 = rng.randint(col_min, col_max)
        if occupied[r0:r0 + ph, c0:c0 + pw].any():
            continue
        target = grid[r0:r0 + ph, c0:c0 + pw]
        non_bg = patch != BG_COLOR
        if np.any(target[non_bg] != BG_COLOR):
            continue
        target[non_bg] = patch[non_bg]
        occupied[r0:r0 + ph, c0:c0 + pw] = True
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG_COLOR, dtype=int)

    legend_side = rng.choice(("left", "right"))
    legend_col = 0 if legend_side == "left" else w - 1
    legend_row = rng.randint(1, h - len(LEGEND_COLORS) - 1)
    for offset, color in enumerate(LEGEND_COLORS):
        grid[legend_row + offset, legend_col] = color

    n_regions = rng.choice(REGION_COUNT_OPTIONS)
    target_counts = _choose_target_counts(rng, n_regions)
    occupied = np.zeros_like(grid, dtype=bool)
    if legend_side == "left":
        occupied[:, :2] = True
    else:
        occupied[:, -2:] = True
    occupied[0, :] = True
    occupied[-1, :] = True

    for count in target_counts:
        patch = _make_region_patch(rng, count)
        if patch is None:
            return None
        if not _place_patch(rng, grid, occupied, patch, legend_side):
            return None

    if _extract_legend(grid, BG_COLOR) != list(LEGEND_COLORS):
        return None

    regions = _enclosed_regions(grid)
    if len(regions) != n_regions:
        return None
    actual_counts = sorted(_trapped_color_count(grid, region, BG_COLOR) for region in regions)
    if actual_counts != sorted(target_counts):
        return None

    if _background_color(grid) != BG_COLOR:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid enclosed-region counting puzzle")
