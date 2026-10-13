from __future__ import annotations

from collections import Counter
import random

import numpy as np


PALETTE = tuple(range(1, 10))
MAX_ATTEMPTS = 600


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _majority_color(values: list[int]) -> int:
    counts = Counter(values)
    best = max(counts.values())
    return min(color for color, cnt in counts.items() if cnt == best)


def _render_pattern(tile: np.ndarray, tile_rows: int, tile_cols: int) -> np.ndarray:
    tile_h, tile_w = tile.shape
    out = np.empty((tile_h * tile_rows, tile_w * tile_cols), dtype=int)
    middle = tile_rows // 2
    for tile_r in range(tile_rows):
        band_tile = np.fliplr(tile) if tile_r == middle else tile
        for tile_c in range(tile_cols):
            r0 = tile_r * tile_h
            c0 = tile_c * tile_w
            out[r0 : r0 + tile_h, c0 : c0 + tile_w] = band_tile
    return out


def _recover_from_candidate(grid: np.ndarray, tile_h: int, tile_w: int) -> tuple[np.ndarray, np.ndarray, int] | None:
    h, w = grid.shape
    tile_rows = h // tile_h
    tile_cols = w // tile_w
    middle = tile_rows // 2

    tile = np.empty((tile_h, tile_w), dtype=int)
    for inner_r in range(tile_h):
        for inner_c in range(tile_w):
            samples: list[int] = []
            for tile_r in range(tile_rows):
                abs_r = tile_r * tile_h + inner_r
                for tile_c in range(tile_cols):
                    shown_c = inner_c if tile_r != middle else tile_w - 1 - inner_c
                    abs_c = tile_c * tile_w + shown_c
                    samples.append(int(grid[abs_r, abs_c]))
            tile[inner_r, inner_c] = _majority_color(samples)

    if np.array_equal(tile, np.fliplr(tile)):
        return None

    repaired = _render_pattern(tile, tile_rows, tile_cols)
    mismatch = int(np.count_nonzero(repaired != grid))
    return tile, repaired, mismatch


def _candidate_outputs(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    by_output: dict[tuple[tuple[int, int], bytes], dict[str, object]] = {}

    for tile_h in _divisors(h):
        tile_rows = h // tile_h
        if tile_rows < 3 or tile_rows % 2 == 0:
            continue
        for tile_w in _divisors(w):
            tile_cols = w // tile_w
            if tile_cols < 2:
                continue
            recovered = _recover_from_candidate(grid, tile_h, tile_w)
            if recovered is None:
                continue
            tile, repaired, mismatch = recovered
            key = (tuple(repaired.shape), repaired.tobytes())
            prev = by_output.get(key)
            if prev is None or mismatch < int(prev["mismatch"]):
                by_output[key] = {
                    "tile": tile,
                    "output": repaired,
                    "mismatch": mismatch,
                    "tile_shape": (tile_h, tile_w),
                }
    return list(by_output.values())


def _repair_without_mirror(grid: np.ndarray, tile_h: int, tile_w: int) -> np.ndarray:
    h, w = grid.shape
    tile_rows = h // tile_h
    tile_cols = w // tile_w
    tile = np.empty((tile_h, tile_w), dtype=int)
    for inner_r in range(tile_h):
        for inner_c in range(tile_w):
            samples = [
                int(grid[tile_r * tile_h + inner_r, tile_c * tile_w + inner_c])
                for tile_r in range(tile_rows)
                for tile_c in range(tile_cols)
            ]
            tile[inner_r, inner_c] = _majority_color(samples)
    out = np.empty_like(grid)
    for tile_r in range(tile_rows):
        for tile_c in range(tile_cols):
            r0 = tile_r * tile_h
            c0 = tile_c * tile_w
            out[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore a mostly tiled grid by inferring the base tile and undoing the mirrored middle tile band.

    Input structure:
    - The full grid is an array of repeated copies of one small tile.
    - All tile rows use the original tile except the middle tile row, whose tiles are left-right mirrored.
    - A small number of pixels have been corrupted, but the repeated structure still dominates.

    Rule:
    - Consider every tile size that evenly partitions the grid into an odd number of tile rows and at least two tile columns.
    - For each candidate, flip the middle tile row back to the tile's original orientation, majority-vote each tile-relative coordinate, and redraw the full grid with the same mirrored-middle construction.
    - Return the unique repaired grid whose redraw disagrees with the input in the fewest cells.

    Key insight:
    - The apparent periodic break in the center is not noise; it is a whole mirrored tile band, so correct recovery requires unmirroring that band before aligning repeated coordinates.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert 0 < g.shape[0] <= 30 and 0 < g.shape[1] <= 30, "grid must fit ARC size limits"

    candidates = _candidate_outputs(g)
    assert candidates, "no valid tile decomposition found"
    best_mismatch = min(int(item["mismatch"]) for item in candidates)
    best = [item for item in candidates if int(item["mismatch"]) == best_mismatch]
    assert len(best) == 1, "expected a unique best repaired output"
    return np.asarray(best[0]["output"], dtype=int).copy()


def _has_smaller_period(tile: np.ndarray, axis: int) -> bool:
    size = tile.shape[axis]
    for period in _divisors(size):
        if period == size:
            continue
        if axis == 0:
            base = tile[:period, :]
            if np.array_equal(tile, np.tile(base, (size // period, 1))):
                return True
        else:
            base = tile[:, :period]
            if np.array_equal(tile, np.tile(base, (1, size // period))):
                return True
    return False


def _ensure_palette_presence(tile: np.ndarray, palette: list[int]) -> None:
    present = {int(x) for x in tile.ravel()}
    free = [(r, c) for r in range(tile.shape[0]) for c in range(tile.shape[1])]
    random.shuffle(free)
    for color in palette:
        if color in present:
            continue
        r, c = free.pop()
        tile[r, c] = color
        present.add(color)


def _sample_tile(tile_h: int, tile_w: int) -> np.ndarray | None:
    color_count = random.randint(3, min(6, len(PALETTE)))
    palette = random.sample(PALETTE, color_count)
    tile = np.array(
        [random.choice(palette) for _ in range(tile_h * tile_w)],
        dtype=int,
    ).reshape(tile_h, tile_w)

    tile[0, 0] = palette[0]
    tile[0, tile_w - 1] = palette[1]
    tile[tile_h - 1, 0] = palette[2]
    if tile_h > 1 and tile_w > 1 and color_count >= 4:
        tile[tile_h - 1, tile_w - 1] = palette[3]

    _ensure_palette_presence(tile, palette)
    if np.array_equal(tile, np.fliplr(tile)):
        return None
    if _has_smaller_period(tile, axis=0) or _has_smaller_period(tile, axis=1):
        return None
    return tile


def _display_coord(tile_r: int, tile_c: int, inner_r: int, inner_c: int, tile_h: int, tile_w: int, middle: int) -> tuple[int, int]:
    abs_r = tile_r * tile_h + inner_r
    shown_c = inner_c if tile_r != middle else tile_w - 1 - inner_c
    abs_c = tile_c * tile_w + shown_c
    return abs_r, abs_c


def _choose_corruption_plan(tile_rows: int, tile_cols: int, tile_h: int, tile_w: int) -> list[tuple[int, int, int, int]]:
    middle = tile_rows // 2
    rel_coords = [(r, c) for r in range(tile_h) for c in range(tile_w)]
    random.shuffle(rel_coords)

    max_corrupt = min(len(rel_coords), 6)
    count = random.randint(2, max_corrupt)
    mode = random.choice(("mixed", "middle_heavy", "outer_only"))

    chosen: list[tuple[int, int, int, int]] = []
    middle_slots = [(middle, tc) for tc in range(tile_cols)]
    outer_slots = [(tr, tc) for tr in range(tile_rows) for tc in range(tile_cols) if tr != middle]

    for idx, (inner_r, inner_c) in enumerate(rel_coords[:count]):
        if mode == "outer_only":
            slots = outer_slots
        elif mode == "middle_heavy":
            slots = middle_slots if idx < max(1, count - 1) else outer_slots
        else:
            if idx == 0:
                slots = middle_slots
            elif idx == 1:
                slots = outer_slots
            else:
                slots = middle_slots + outer_slots
        tile_r, tile_c = random.choice(slots)
        chosen.append((tile_r, tile_c, inner_r, inner_c))
    return chosen


def _corrupt_grid(clean: np.ndarray, tile_h: int, tile_w: int) -> np.ndarray:
    h, w = clean.shape
    tile_rows = h // tile_h
    tile_cols = w // tile_w
    middle = tile_rows // 2
    corrupted = clean.copy()

    for tile_r, tile_c, inner_r, inner_c in _choose_corruption_plan(tile_rows, tile_cols, tile_h, tile_w):
        abs_r, abs_c = _display_coord(tile_r, tile_c, inner_r, inner_c, tile_h, tile_w, middle)
        current = int(clean[abs_r, abs_c])
        choices = [color for color in PALETTE if color != current]
        corrupted[abs_r, abs_c] = random.choice(choices)
    return corrupted


def _valid_case(inp: np.ndarray, clean: np.ndarray, tile_h: int, tile_w: int) -> bool:
    if np.array_equal(inp, clean):
        return False
    try:
        repaired = main(inp)
    except Exception:
        return False
    if not np.array_equal(repaired, clean):
        return False

    candidates = _candidate_outputs(inp)
    best_mismatch = min(int(item["mismatch"]) for item in candidates)
    best = [item for item in candidates if int(item["mismatch"]) == best_mismatch]
    if len(best) != 1:
        return False
    if not np.array_equal(np.asarray(best[0]["output"], dtype=int), clean):
        return False

    naive = _repair_without_mirror(inp, tile_h, tile_w)
    if np.array_equal(naive, clean):
        return False
    if np.count_nonzero(naive != clean) < 2:
        return False

    if np.count_nonzero(inp != clean) > clean.size // 4:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        tile_h = random.randint(2, 5)
        tile_w = random.randint(2, 6)
        tile_rows = random.choice([3, 5])
        tile_cols = random.randint(2, min(5, 30 // tile_w))
        if tile_h * tile_rows > 30 or tile_w * tile_cols > 30:
            continue

        tile = _sample_tile(tile_h, tile_w)
        if tile is None:
            continue
        clean = _render_pattern(tile, tile_rows, tile_cols)
        inp = _corrupt_grid(clean, tile_h, tile_w)
        if _valid_case(inp, clean, tile_h, tile_w):
            return inp
    raise RuntimeError("failed to generate a valid mirrored-band repair input")
