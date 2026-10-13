from __future__ import annotations

import random
from collections import Counter

import numpy as np


GREY = 5
GEN_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
PERIOD_CHOICES = (3, 4)
FOREGROUND_COUNT_CHOICES = (2, 3)
HOLE_COUNT_RANGE = (5, 10)
GRID_SIZE_RANGE = (9, 14)
MAX_GENERATION_ATTEMPTS = 500


def _tile_grid(tile: np.ndarray, h: int, w: int) -> np.ndarray:
    rows = np.arange(h)[:, None] % tile.shape[0]
    cols = np.arange(w)[None, :] % tile.shape[1]
    return tile[rows, cols]


def _color_order(grid: np.ndarray) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    for value in grid.flat:
        color = int(value)
        if color != 0 and color not in seen:
            seen.add(color)
            order.append(color)
    return order


def _infer_periodic_tile(grid: np.ndarray) -> tuple[int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    for p in range(1, min(h, w) + 1):
        tile = np.full((p, p), -1, dtype=int)
        ok = True
        for r in range(h):
            for c in range(w):
                value = int(g[r, c])
                if value == 0:
                    continue
                rr, cc = r % p, c % p
                if tile[rr, cc] == -1:
                    tile[rr, cc] = value
                elif tile[rr, cc] != value:
                    ok = False
                    break
            if not ok:
                break
        if ok and np.all(tile != -1):
            repaired = _tile_grid(tile, h, w)
            if np.all((g == 0) | (g == repaired)):
                return p, tile
    raise AssertionError("no complete periodic tile fits the visible cells")


def _find_guide_frame(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    grey_coords = np.argwhere(grid == GREY)
    assert grey_coords.size > 0, "expected a grey guide frame"
    r0, c0 = grey_coords.min(axis=0)
    r1, c1 = grey_coords.max(axis=0)
    r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
    assert r1 == grid.shape[0] - 1 and c1 == grid.shape[1] - 1
    frame = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert frame.shape[0] == 3 and frame.shape[1] >= 4
    assert np.all(frame[0, :] == GREY) and np.all(frame[-1, :] == GREY)
    assert np.all(frame[:, 0] == GREY) and np.all(frame[:, -1] == GREY)
    interior = frame[1:-1, 1:-1]
    assert interior.shape[0] == 1
    guide = interior[0].astype(int)
    assert len(guide) >= 2
    assert len(set(map(int, guide))) == len(guide)
    assert all(color not in (0, GREY) for color in guide)
    grey_mask = np.zeros_like(grid, dtype=bool)
    grey_mask[r0 : r1 + 1, c0 : c1 + 1] = frame == GREY
    assert np.array_equal(grid == GREY, grey_mask)
    return r0, c0, guide


def _recolor_by_order(grid: np.ndarray, color_order: list[int], guide: list[int] | np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for src, dst in zip(color_order, guide):
        out[grid == src] = int(dst)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the periodic table, then recolor each repaired foreground color by the grey-framed guide order.

    Input structure:
    - The lower-right corner contains one grey rectangular frame whose 1xK interior is a color guide strip.
    - The top-left region outside that frame is the main grid: a square-periodic color table with black holes.
    - Each non-black periodic residue remains visible at least once elsewhere, so the smallest repeating tile can be recovered.

    Rule:
    - Isolate the grey-framed guide strip and discard the frame from the output.
    - On the main grid, infer the smallest period p such that visible non-black cells agree by (row mod p, col mod p), then repair every black hole from that p x p tile.
    - Scan the repaired main grid in row-major order, record the first appearance order of its distinct non-black colors, and globally replace the i-th color with the i-th guide color.

    Key insight:
    - The guide does not name fixed quadrants or fixed source colors; it names the repaired color classes in first-appearance order, so the periodic repair must happen before recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    guide_r0, guide_c0, guide = _find_guide_frame(g)
    main_grid = g[:guide_r0, :guide_c0]
    assert main_grid.size > 0

    _period, tile = _infer_periodic_tile(main_grid)
    repaired = _tile_grid(tile, *main_grid.shape)
    color_order = _color_order(repaired)
    assert len(color_order) == len(guide)
    return _recolor_by_order(repaired, color_order, guide)


def _sample_tile(period: int, colors: list[int]) -> np.ndarray | None:
    for _ in range(200):
        tile = np.random.choice(colors, size=(period, period), replace=True)
        counts = Counter(map(int, tile.flat))
        if set(counts) != set(colors):
            continue
        if min(counts.values()) < 2:
            continue
        sample = _tile_grid(tile, period * 3 + 1, period * 3 + 2)
        inferred_period, _ = _infer_periodic_tile(sample)
        if inferred_period != period:
            continue
        return tile.astype(int)
    return None


def _positions_by_color(grid: np.ndarray, colors: list[int]) -> dict[int, list[tuple[int, int]]]:
    positions: dict[int, list[tuple[int, int]]] = {}
    for color in colors:
        coords = np.argwhere(grid == color)
        positions[color] = [tuple(map(int, coord)) for coord in coords]
    return positions


def _forced_holes(full_grid: np.ndarray, color_order: list[int]) -> list[tuple[int, int]] | None:
    positions = _positions_by_color(full_grid, color_order)
    best: list[tuple[int, int]] | None = None
    for i in range(len(color_order) - 1):
        earlier = color_order[i]
        for later in color_order[i + 1 :]:
            cutoff = positions[later][0]
            forced = [pos for pos in positions[earlier] if pos < cutoff]
            if 1 <= len(forced) <= HOLE_COUNT_RANGE[1] - 1:
                if best is None or len(forced) < len(best):
                    best = forced
    return best


def _sample_holes(
    full_grid: np.ndarray,
    period: int,
    forced_holes: list[tuple[int, int]],
    target_count: int,
) -> set[tuple[int, int]] | None:
    h, w = full_grid.shape
    residue_counts = Counter((r % period, c % period) for r in range(h) for c in range(w))
    color_counts = Counter(map(int, full_grid.flat))
    holes: set[tuple[int, int]] = set()

    def add_hole(coord: tuple[int, int]) -> bool:
        r, c = coord
        residue = (r % period, c % period)
        color = int(full_grid[r, c])
        if residue_counts[residue] <= 1 or color_counts[color] <= 1:
            return False
        holes.add(coord)
        residue_counts[residue] -= 1
        color_counts[color] -= 1
        return True

    for coord in forced_holes:
        if not add_hole(coord):
            return None

    candidates = [(r, c) for r in range(h) for c in range(w) if (r, c) not in holes]
    random.shuffle(candidates)
    for coord in candidates:
        if len(holes) >= target_count:
            break
        add_hole(coord)
    if len(holes) != target_count:
        return None
    return holes


def _build_input_from_parts(main_grid: np.ndarray, guide_colors: list[int]) -> np.ndarray:
    h, w = main_grid.shape
    k = len(guide_colors)
    out = np.zeros((h + 3, w + k + 2), dtype=int)
    out[:h, :w] = main_grid
    out[h:, w:] = GREY
    out[h + 1, w + 1 : w + 1 + k] = np.array(guide_colors, dtype=int)
    return out


def _generate_case() -> np.ndarray | None:
    period = random.choice(PERIOD_CHOICES)
    color_count = random.choice(FOREGROUND_COUNT_CHOICES)
    foreground_colors = random.sample(GEN_COLORS, color_count)
    guide_colors = random.sample([c for c in GEN_COLORS if c not in foreground_colors], color_count)

    tile = _sample_tile(period, foreground_colors)
    if tile is None:
        return None

    min_size = max(GRID_SIZE_RANGE[0], period * 2 + 1)
    max_size = GRID_SIZE_RANGE[1]
    h = random.randint(min_size, max_size)
    w = random.randint(min_size, max_size)
    full_grid = _tile_grid(tile, h, w)

    full_order = _color_order(full_grid)
    if len(full_order) != color_count:
        return None

    forced = _forced_holes(full_grid, full_order)
    if forced is None:
        return None

    target_holes = random.randint(max(HOLE_COUNT_RANGE[0], len(forced)), HOLE_COUNT_RANGE[1])
    holes = _sample_holes(full_grid, period, forced, target_holes)
    if holes is None:
        return None

    main_grid = full_grid.copy()
    for r, c in holes:
        main_grid[r, c] = 0

    try:
        inferred_period, inferred_tile = _infer_periodic_tile(main_grid)
    except AssertionError:
        return None
    if inferred_period != period:
        return None
    repaired = _tile_grid(inferred_tile, h, w)
    if not np.array_equal(repaired, full_grid):
        return None

    visible_order = _color_order(main_grid)
    if visible_order == full_order or len(visible_order) != len(full_order):
        return None

    correct = _recolor_by_order(repaired, full_order, guide_colors)
    shortcut = _recolor_by_order(repaired, visible_order, guide_colors)
    if np.array_equal(correct, shortcut):
        return None

    return _build_input_from_parts(main_grid, guide_colors)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case
    raise RuntimeError("failed to generate a valid input")
