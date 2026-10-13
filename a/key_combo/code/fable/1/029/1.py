from __future__ import annotations

import random

import numpy as np


PALETTE = tuple(range(1, 10))
COPY_OPTIONS = (3, 5)
GEN_RETRY_LIMIT = 400


def _mirror_coord(local_r: int, local_c: int, tile_h: int, tile_w: int, tile_r: int, tile_c: int) -> tuple[int, int]:
    """Map a tile-local coordinate through the tile's row/col parity mirrors."""
    if tile_r % 2 == 1:
        local_r = tile_h - 1 - local_r
    if tile_c % 2 == 1:
        local_c = tile_w - 1 - local_c
    return int(local_r), int(local_c)


def _tile_grid(mother: np.ndarray, row_copies: int, col_copies: int) -> np.ndarray:
    tile_h, tile_w = mother.shape
    grid = np.zeros((tile_h * row_copies, tile_w * col_copies), dtype=int)
    for tile_r in range(row_copies):
        row_view = mother[::-1, :] if tile_r % 2 == 1 else mother
        for tile_c in range(col_copies):
            block = row_view[:, ::-1] if tile_c % 2 == 1 else row_view
            r0 = tile_r * tile_h
            c0 = tile_c * tile_w
            grid[r0 : r0 + tile_h, c0 : c0 + tile_w] = block
    return grid


def _analyze_candidate(grid: np.ndarray, tile_h: int, tile_w: int) -> dict | None:
    height, width = map(int, grid.shape)
    if height % tile_h != 0 or width % tile_w != 0:
        return None

    row_copies = height // tile_h
    col_copies = width // tile_w
    if row_copies < 2 or col_copies < 2:
        return None

    copies = row_copies * col_copies
    counts = [[{} for _ in range(tile_w)] for _ in range(tile_h)]
    for r in range(height):
        tile_r, local_r = divmod(r, tile_h)
        for c in range(width):
            tile_c, local_c = divmod(c, tile_w)
            base_r, base_c = _mirror_coord(local_r, local_c, tile_h, tile_w, tile_r, tile_c)
            color = int(grid[r, c])
            bucket = counts[base_r][base_c]
            bucket[color] = bucket.get(color, 0) + 1

    mother = np.zeros((tile_h, tile_w), dtype=int)
    mother_colors: set[int] = set()
    anomaly_color: int | None = None
    anomaly_coords: list[tuple[int, int]] = []

    for base_r in range(tile_h):
        for base_c in range(tile_w):
            bucket = counts[base_r][base_c]
            if len(bucket) == 1:
                color, count = next(iter(bucket.items()))
                if count != copies:
                    return None
                mother[base_r, base_c] = int(color)
                mother_colors.add(int(color))
                continue

            if len(bucket) != 2:
                return None

            ranked = sorted(bucket.items(), key=lambda item: item[1], reverse=True)
            (clean_color, clean_count), (odd_color, odd_count) = ranked
            if clean_count != copies - 1 or odd_count != 1:
                return None

            if anomaly_color is None:
                anomaly_color = int(odd_color)
            elif anomaly_color != int(odd_color):
                return None

            mother[base_r, base_c] = int(clean_color)
            mother_colors.add(int(clean_color))
            anomaly_coords.append((base_r, base_c))

    if anomaly_color is None:
        return None
    if anomaly_color in mother_colors:
        return None
    if len(anomaly_coords) != len(set(anomaly_coords)):
        return None

    output = mother.copy()
    for base_r, base_c in anomaly_coords:
        output[base_r, base_c] = anomaly_color

    return {
        "shape": (int(tile_h), int(tile_w)),
        "mother": mother,
        "anomaly_color": int(anomaly_color),
        "anomaly_coords": tuple(anomaly_coords),
        "output": output,
    }


def _valid_candidates(grid: np.ndarray) -> list[dict]:
    height, width = map(int, grid.shape)
    candidates: list[dict] = []
    for tile_h in range(1, height):
        if height % tile_h != 0:
            continue
        for tile_w in range(1, width):
            if width % tile_w != 0:
                continue
            candidate = _analyze_candidate(grid, tile_h, tile_w)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _best_candidate(grid: np.ndarray) -> dict:
    candidates = _valid_candidates(grid)
    assert candidates, "expected at least one mirror-tiling candidate"

    candidates.sort(key=lambda item: (item["shape"][0] * item["shape"][1], item["shape"][0], item["shape"][1]))
    best = candidates[0]
    best_area = best["shape"][0] * best["shape"][1]
    tied = [item for item in candidates if item["shape"][0] * item["shape"][1] == best_area]
    assert len(tied) == 1, "expected a unique smallest valid mother tile"
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold anomaly pixels from an alternating mirror tiling back onto the mother tile.

    Input structure:
    - The full grid is tiled by copies of the top-left mother tile.
    - Copies in odd tile columns are mirrored left-right, copies in odd tile rows are mirrored up-down, and odd/odd copies are mirrored both ways.
    - A few cells have been overwritten by one anomaly color that does not belong to the clean mother tile.

    Rule:
    - Infer the mother tile size and clean contents from the repeated mirrored copies.
    - For each anomaly pixel, undo the mirrors of its own copy so the pixel folds back to its mother-tile coordinate.
    - Output the mother tile, then overwrite those folded coordinates with the anomaly color.

    Key insight:
    - Simple modulo folding is wrong because mirrored copies reverse their local coordinates; each anomaly must be sent home through the inverse mirror of the specific copy that contains it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _best_candidate(g)["output"]


def _random_mother(tile_h: int, tile_w: int, palette: list[int]) -> np.ndarray | None:
    mother = np.random.choice(np.array(palette, dtype=int), size=(tile_h, tile_w))
    if len(set(map(int, mother.flat))) < min(3, len(palette)):
        return None
    if np.array_equal(mother, mother[::-1, :]):
        return None
    if np.array_equal(mother, mother[:, ::-1]):
        return None
    return mother.astype(int)


def _choose_anomaly_tiles(row_copies: int, col_copies: int, count: int) -> list[tuple[int, int]]:
    all_tiles = [(tile_r, tile_c) for tile_r in range(row_copies) for tile_c in range(col_copies)]
    chosen: list[tuple[int, int]] = []
    parity_groups = {
        parity: [tile for tile in all_tiles if (tile[0] % 2, tile[1] % 2) == parity]
        for parity in ((0, 0), (0, 1), (1, 0), (1, 1))
    }
    if count >= 4:
        for parity in ((0, 0), (0, 1), (1, 0), (1, 1)):
            chosen.append(random.choice(parity_groups[parity]))

    remaining = [tile for tile in all_tiles if tile not in chosen]
    random.shuffle(remaining)
    chosen.extend(remaining[: count - len(chosen)])
    return chosen


def _make_case() -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
    row_copies = random.choice(COPY_OPTIONS)
    col_copies = random.choice(COPY_OPTIONS)
    max_tile_h = min(8, 30 // row_copies)
    max_tile_w = min(8, 30 // col_copies)
    if max_tile_h < 3 or max_tile_w < 3:
        return None

    tile_h = random.randint(3, max_tile_h)
    tile_w = random.randint(3, max_tile_w)
    palette_size = random.randint(3, 4)
    palette = random.sample(PALETTE, palette_size)
    mother = _random_mother(tile_h, tile_w, palette)
    if mother is None:
        return None

    anomaly_choices = [color for color in PALETTE if color not in palette]
    anomaly_color = random.choice(anomaly_choices)

    grid = _tile_grid(mother, row_copies, col_copies)

    max_anomalies = min(6, tile_h * tile_w, row_copies * col_copies)
    anomaly_count = random.randint(2, max_anomalies)
    anomaly_coords = random.sample([(r, c) for r in range(tile_h) for c in range(tile_w)], anomaly_count)
    anomaly_tiles = _choose_anomaly_tiles(row_copies, col_copies, anomaly_count)

    expected = mother.copy()
    for (base_r, base_c), (tile_r, tile_c) in zip(anomaly_coords, anomaly_tiles):
        local_r, local_c = _mirror_coord(base_r, base_c, tile_h, tile_w, tile_r, tile_c)
        grid[tile_r * tile_h + local_r, tile_c * tile_w + local_c] = anomaly_color
        expected[base_r, base_c] = anomaly_color

    best = _best_candidate(grid)
    if best["shape"] != (tile_h, tile_w):
        return None
    if not np.array_equal(best["output"], expected):
        return None
    return grid, expected, (tile_h, tile_w)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        case = _make_case()
        if case is None:
            continue
        grid, expected, _shape = case
        actual = main(grid)
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid mirror-folding sample")
