import random
from collections import Counter, deque

import numpy as np


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(pts)
    return parts


def _parse_legend(grid: np.ndarray, bg: int) -> tuple[int, dict[int, np.ndarray]]:
    h, w = grid.shape
    for c0 in range(w - 2, -1, -1):
        strip = grid[:, c0 : c0 + 2]
        if strip.shape[1] != 2:
            continue

        entries: list[tuple[int, int, np.ndarray]] = []
        r = 0
        while r <= h - 2:
            block = strip[r : r + 2]
            if (
                block.shape == (2, 2)
                and np.all(block != bg)
                and int(block[0, 0]) == int(block[1, 0])
            ):
                entries.append((r, int(block[0, 0]), block[:, 1:].copy()))
                r += 2
                while r < h and np.all(strip[r] == bg):
                    r += 1
                continue
            r += 1

        if not entries:
            continue

        covered = np.zeros(h, dtype=bool)
        mapping: dict[int, np.ndarray] = {}
        ok = True
        for r0, marker, tile in entries:
            if marker in mapping:
                ok = False
                break
            mapping[marker] = np.asarray(tile, dtype=int)
            covered[r0 : r0 + 2] = True

        if not ok:
            continue

        for rr in range(h):
            if np.any(strip[rr] != bg) and not bool(covered[rr]):
                ok = False
                break
        if ok:
            return c0, mapping

    raise AssertionError("failed to locate the right-edge legend")


def _find_regions(board: np.ndarray, bg: int, wall: int) -> list[tuple[int, int, int, int, int]]:
    h, w = board.shape
    passable = board != wall
    regions: list[tuple[int, int, int, int, int]] = []

    for comp in _components(passable):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue

        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        if not np.all(board[r0 : r1 + 1, c0 : c1 + 1] != wall):
            continue
        if not (
            np.all(board[r0 - 1, c0 - 1 : c1 + 2] == wall)
            and np.all(board[r1 + 1, c0 - 1 : c1 + 2] == wall)
            and np.all(board[r0 - 1 : r1 + 2, c0 - 1] == wall)
            and np.all(board[r0 - 1 : r1 + 2, c1 + 1] == wall)
        ):
            continue

        marker_colors = {int(board[r, c]) for r, c in comp if int(board[r, c]) != bg}
        if len(marker_colors) != 1:
            continue
        marker = next(iter(marker_colors))
        regions.append((r0, c0, r1, c1, marker))

    return regions


def _tile_patch(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    tile_h, tile_w = tile.shape
    rr = np.arange(height)[:, None] % tile_h
    cc = np.arange(width)[None, :] % tile_w
    return tile[rr, cc]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each walled rectangle with the striped pattern keyed by its internal marker color.

    Input structure:
    - The background is the dominant color.
    - Several rectangular chambers are enclosed by one common wall color, and each chamber contains 1-3 scattered marker pixels of a single marker color.
    - The far-right edge contains a vertical legend made of 2x2 blocks; in each block the left column is a solid marker color and the right column stores a two-row fill tile.

    Rule:
    - Read the right-edge legend to build a mapping from marker color to its 2-row tile.
    - Find every enclosed rectangular chamber in the board area left of the legend and read the unique non-background marker color inside it.
    - Starting from the chamber interior's top-left cell, periodically tile the corresponding legend pattern across the full interior, overwriting both background and marker pixels.
    - Leave walls, legend cells, and exterior background unchanged.

    Key insight:
    - The scattered marker pixels are only lookup keys; the actual fill comes from the legend's two-row pattern, which must be phase-aligned to each chamber's own top-left corner.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    legend_left, legend_map = _parse_legend(g, bg)

    board = g[:, :legend_left]
    board_colors = [int(v) for v in board.ravel() if int(v) != bg]
    if not board_colors:
        return g.copy()
    wall = int(Counter(board_colors).most_common(1)[0][0])

    out = g.copy()
    for r0, c0, r1, c1, marker in _find_regions(board, bg, wall):
        tile = legend_map.get(marker)
        if tile is None:
            continue
        out[r0 : r1 + 1, c0 : c1 + 1] = _tile_patch(tile, r1 - r0 + 1, c1 - c0 + 1)
    return out


# --------------- generator ---------------

PALETTE = tuple(range(10))
LEGEND_WIDTH = 2
SPACER_CHOICES = (1, 2)
REGION_COUNT_CHOICES = (1, 2, 3, 4)
HEIGHT_RANGE = (12, 28)
WIDTH_RANGE = (16, 30)
MARKER_COUNT_CHOICES = (1, 2, 3)
LEGEND_GAP_CHOICES = (0, 1)
MAX_GENERATION_ATTEMPTS = 400
MAX_PLACEMENT_ATTEMPTS = 200


def _draw_frame(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int, wall: int) -> None:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top, left : right + 1] = wall
    grid[bottom, left : right + 1] = wall
    grid[top : bottom + 1, left] = wall
    grid[top : bottom + 1, right] = wall


def _place_regions(
    grid: np.ndarray,
    board_w: int,
    wall: int,
    markers: list[int],
    rng: random.Random,
) -> bool:
    h = grid.shape[0]
    occupied = np.zeros((h, board_w), dtype=bool)

    for marker in markers:
        placed = False
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            max_inner_h = min(7, h - 4)
            max_inner_w = min(8, board_w - 4)
            if max_inner_h < 2 or max_inner_w < 2:
                return False

            inner_h = rng.randint(2, max_inner_h)
            inner_w = rng.randint(2, max_inner_w)
            if inner_h * inner_w < 4:
                continue

            outer_h = inner_h + 2
            outer_w = inner_w + 2
            top = rng.randint(1, h - outer_h - 1)
            left = rng.randint(1, board_w - outer_w - 1)

            hr0 = max(0, top - 1)
            hr1 = min(h, top + outer_h + 1)
            hc0 = max(0, left - 1)
            hc1 = min(board_w, left + outer_w + 1)
            if np.any(occupied[hr0:hr1, hc0:hc1]):
                continue

            _draw_frame(grid, top, left, inner_h, inner_w, wall)
            occupied[hr0:hr1, hc0:hc1] = True

            inside = [(r, c) for r in range(top + 1, top + inner_h + 1) for c in range(left + 1, left + inner_w + 1)]
            max_markers = min(3, len(inside) - 1)
            if max_markers < 1:
                return False
            marker_count = rng.randint(1, max_markers)
            for r, c in rng.sample(inside, marker_count):
                grid[r, c] = marker
            placed = True
            break

        if not placed:
            return False
    return True


def _place_legend(
    grid: np.ndarray,
    marker_order: list[int],
    pattern_map: dict[int, np.ndarray],
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    gaps = [rng.choice(LEGEND_GAP_CHOICES) for _ in range(max(0, len(marker_order) - 1))]
    total_h = 2 * len(marker_order) + sum(gaps)
    if total_h > h:
        return False

    top = rng.randint(0, h - total_h)
    col = w - LEGEND_WIDTH
    row = top
    for i, marker in enumerate(marker_order):
        grid[row : row + 2, col] = marker
        grid[row : row + 2, col + 1 : col + 2] = pattern_map[marker]
        row += 2
        if i < len(gaps):
            row += gaps[i]
    return True


def _valid_candidate(grid: np.ndarray, bg: int, region_count: int) -> bool:
    counts = Counter(map(int, grid.ravel()))
    if int(counts.most_common(1)[0][0]) != bg:
        return False
    try:
        out = main(grid)
    except Exception:
        return False
    if out.shape != grid.shape or not np.any(out != grid):
        return False

    legend_left, _ = _parse_legend(grid, bg)
    board = grid[:, :legend_left]
    wall_candidates = [int(v) for v in board.ravel() if int(v) != bg]
    if not wall_candidates:
        return False
    wall = int(Counter(wall_candidates).most_common(1)[0][0])
    return len(_find_regions(board, bg, wall)) == region_count


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    region_count = rng.choice(REGION_COUNT_CHOICES)
    spacer = rng.choice(SPACER_CHOICES)

    min_h = max(HEIGHT_RANGE[0], 2 * region_count + 2)
    h = rng.randint(min_h, HEIGHT_RANGE[1])
    w = rng.randint(WIDTH_RANGE[0], WIDTH_RANGE[1])
    board_w = w - spacer - LEGEND_WIDTH
    if board_w < 8:
        return None

    bg, wall = rng.sample(PALETTE, 2)
    available = [c for c in PALETTE if c not in {bg, wall}]
    if len(available) < region_count:
        return None
    marker_colors = rng.sample(available, region_count)

    pattern_pool = [c for c in available if c not in marker_colors]
    if len(pattern_pool) < 2:
        pattern_pool = available[:]

    pattern_map: dict[int, np.ndarray] = {}
    for marker in marker_colors:
        choices = [c for c in pattern_pool if c != marker]
        if len(choices) < 2:
            choices = [c for c in available if c != marker]
        a, b = rng.sample(choices, 2)
        pattern_map[marker] = np.array([[a], [b]], dtype=int)

    grid = np.full((h, w), bg, dtype=int)
    if not _place_regions(grid, board_w, wall, marker_colors, rng):
        return None

    legend_order = marker_colors[:]
    rng.shuffle(legend_order)
    if not _place_legend(grid, legend_order, pattern_map, rng):
        return None

    if not _valid_candidate(grid, bg, region_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to produce a valid sample")
