import random
from collections import deque
from itertools import combinations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
BROWN = 9

SEED_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE)
RESULT_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, CYAN, BROWN)
ALL_SEED_PAIRS = list(combinations(SEED_COLORS, 2))

GEN_RETRIES = 400
PLACEMENT_RETRIES = 150


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(cells)
    return parts


def _paint_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _frame_bbox_from_component(grid: np.ndarray, cells: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 4 or width < 4:
        return None
    if len(cells) != 2 * (height + width) - 4:
        return None
    block = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(block[0, :] == GREY) or not np.all(block[-1, :] == GREY):
        return None
    if not np.all(block[:, 0] == GREY) or not np.all(block[:, -1] == GREY):
        return None
    if np.any(block[1:-1, 1:-1] == GREY):
        return None
    return r0, c0, r1, c1


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for cells in _components(grid == GREY):
        bbox = _frame_bbox_from_component(grid, cells)
        if bbox is not None:
            frames.append(bbox)
    assert frames, "expected at least one grey rectangular frame"
    return frames


def _find_legend(
    grid: np.ndarray, frames: list[tuple[int, int, int, int]]
) -> tuple[tuple[int, int, int, int], list[tuple[int, int, int]]]:
    outside = np.ones_like(grid, dtype=bool)
    for r0, c0, r1, c1 in frames:
        outside[r0 : r1 + 1, c0 : c1 + 1] = False

    candidates = []
    for cells in _components((grid != BLACK) & outside):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if c1 - c0 + 1 != 3:
            continue
        if len(cells) != (r1 - r0 + 1) * 3:
            continue
        if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] == BLACK):
            continue
        candidates.append((r0, c0, r1, c1))

    assert candidates, "expected a filled 3-column legend outside the frames"
    legend_bbox = min(candidates, key=lambda bbox: (bbox[0], -bbox[3]))
    r0, c0, r1, c1 = legend_bbox
    entries: list[tuple[int, int, int]] = []
    for r in range(r0, r1 + 1):
        left, middle, result = map(int, grid[r, c0 : c1 + 1])
        entries.append((left, middle, result))
    return legend_bbox, entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey rectangle border by looking up its two seed colors in the top-right legend.

    Input structure:
    - The background is black.
    - Several disconnected one-cell-thick grey rectangular frames appear in the grid; each frame encloses a black interior.
    - Inside every frame there are exactly two singleton seed pixels of different colors.
    - A filled 3-column legend block sits near the top-right corner; each row stores two seed colors and one output color.

    Rule:
    - Detect every grey rectangular frame and read the two non-black seed colors inside it.
    - Find the legend row whose first two cells contain the same unordered pair of seed colors.
    - Repaint the entire grey border of that frame with the legend row's third color.
    - Keep the two seed pixels unchanged and erase the whole legend block to black.

    Key insight:
    - The legend is a separate color-pair lookup table, so the task is solved by matching each region's two interior seeds to one legend row and applying that row's result color to the border.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_frames(g)
    legend_bbox, legend_entries = _find_legend(g, frames)
    lookup = {frozenset((a, b)): result for a, b, result in legend_entries}

    out = g.copy()
    lr0, lc0, lr1, lc1 = legend_bbox
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = BLACK

    for bbox in frames:
        r0, c0, r1, c1 = bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_values = list(map(int, interior[interior != BLACK]))
        assert len(seed_values) == 2, "each frame must contain exactly two seed pixels"
        assert seed_values[0] != seed_values[1], "the two seeds must have different colors"
        result = lookup[frozenset(seed_values)]
        _paint_border(out, bbox, result)

    return out


def _expanded_slice(start: int, end: int, limit: int) -> slice:
    return slice(max(0, start - 1), min(limit, end + 2))


def _can_place(occupied: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    h, w = occupied.shape
    r0, c0, r1, c1 = bbox
    return not occupied[_expanded_slice(r0, r1, h), _expanded_slice(c0, c1, w)].any()


def _claim(occupied: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    h, w = occupied.shape
    r0, c0, r1, c1 = bbox
    occupied[_expanded_slice(r0, r1, h), _expanded_slice(c0, c1, w)] = True


def _pick_two_positions(rng: random.Random, cells: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    first = rng.choice(cells)
    spaced = [cell for cell in cells if cell != first and abs(cell[0] - first[0]) + abs(cell[1] - first[1]) >= 2]
    second = rng.choice(spaced or [cell for cell in cells if cell != first])
    return first, second


def _render_expected(
    grid: np.ndarray,
    legend_bbox: tuple[int, int, int, int],
    regions: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    lr0, lc0, lr1, lc1 = legend_bbox
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = BLACK
    for region in regions:
        _paint_border(out, region["bbox"], region["result"])
    return out


def _build_input(rng: random.Random) -> np.ndarray | None:
    n_regions = rng.randint(2, 5)
    n_pairs = rng.randint(1, min(4, n_regions))
    pair_catalog = rng.sample(ALL_SEED_PAIRS, n_pairs)
    result_catalog = rng.sample(RESULT_COLORS, n_pairs)
    lookup = {frozenset(pair): result for pair, result in zip(pair_catalog, result_catalog)}

    assignments = pair_catalog[:]
    while len(assignments) < n_regions:
        assignments.append(rng.choice(pair_catalog))
    rng.shuffle(assignments)

    min_height = max(10, n_pairs + 5, 6 + 2 * ((n_regions + 1) // 2))
    min_width = max(14, 10 + 2 * min(n_regions, 4))
    h = rng.randint(min_height, 30)
    w = rng.randint(min_width, 30)

    legend_top = rng.randint(0, min(2, h - n_pairs))
    right_margin = rng.randint(0, 1)
    legend_left = w - 3 - right_margin
    if legend_left < 0:
        return None
    legend_bbox = (legend_top, legend_left, legend_top + n_pairs - 1, legend_left + 2)

    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    _claim(occupied, legend_bbox)

    legend_rows = list(zip(pair_catalog, result_catalog))
    rng.shuffle(legend_rows)
    for row_offset, (pair, result) in enumerate(legend_rows):
        colors = list(pair)
        if rng.random() < 0.5:
            colors[0], colors[1] = colors[1], colors[0]
        grid[legend_top + row_offset, legend_left : legend_left + 3] = (colors[0], colors[1], result)

    regions: list[dict[str, object]] = []
    for pair in assignments:
        placed = False
        max_rect_h = 9 if n_regions <= 3 else 7
        max_rect_w = 10 if n_regions <= 3 else 8
        max_rect_h = min(max_rect_h, h - 2)
        max_rect_w = min(max_rect_w, w - 2)
        if max_rect_h < 4 or max_rect_w < 4:
            return None

        for _ in range(PLACEMENT_RETRIES):
            rect_h = rng.randint(4, max_rect_h)
            rect_w = rng.randint(4, max_rect_w)
            r0 = rng.randint(1, h - rect_h - 1)
            c0 = rng.randint(1, w - rect_w - 1)
            bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            if not _can_place(occupied, bbox):
                continue

            _paint_border(grid, bbox, GREY)
            interior_cells = [(r, c) for r in range(r0 + 1, r0 + rect_h - 1) for c in range(c0 + 1, c0 + rect_w - 1)]
            first_seed, second_seed = _pick_two_positions(rng, interior_cells)
            grid[first_seed] = pair[0]
            grid[second_seed] = pair[1]
            _claim(occupied, bbox)
            regions.append({"bbox": bbox, "pair": pair, "result": lookup[frozenset(pair)]})
            placed = True
            break

        if not placed:
            return None

    expected = _render_expected(grid, legend_bbox, regions)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _build_input(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
