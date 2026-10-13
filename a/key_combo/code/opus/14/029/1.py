import random
from collections import Counter

import numpy as np


BG = 0
FG = 1
SEP = 5
TILE = 3
ROW_COUNT = 3
COL_COUNT = 4
LATTICE_H = ROW_COUNT * (TILE + 1) + 1
LATTICE_W = COL_COUNT * (TILE + 1) + 1
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _find_lattice(grid: np.ndarray) -> tuple[list[int], list[int]]:
    sep_coords = np.argwhere(grid == SEP)
    assert sep_coords.size > 0, "expected grey separator lattice"
    r0, c0 = sep_coords.min(axis=0)
    r1, c1 = sep_coords.max(axis=0)

    row_lines = [r for r in range(int(r0), int(r1) + 1) if np.all(grid[r, int(c0) : int(c1) + 1] == SEP)]
    col_lines = [c for c in range(int(c0), int(c1) + 1) if np.all(grid[int(r0) : int(r1) + 1, c] == SEP)]

    assert len(row_lines) == ROW_COUNT + 1, "expected 4 lattice rows"
    assert len(col_lines) == COL_COUNT + 1, "expected 5 lattice columns"
    assert all(row_lines[i + 1] - row_lines[i] == TILE + 1 for i in range(ROW_COUNT)), "bad row spacing"
    assert all(col_lines[i + 1] - col_lines[i] == TILE + 1 for i in range(COL_COUNT)), "bad col spacing"
    return row_lines, col_lines


def _extract_tiles(grid: np.ndarray, row_lines: list[int], col_lines: list[int]) -> list[list[np.ndarray]]:
    rows: list[list[np.ndarray]] = []
    for i in range(ROW_COUNT):
        tiles = []
        for j in range(COL_COUNT):
            tile = grid[row_lines[i] + 1 : row_lines[i + 1], col_lines[j] + 1 : col_lines[j + 1]].copy()
            assert tile.shape == (TILE, TILE), "expected 3x3 tile"
            tiles.append(tile)
        rows.append(tiles)
    return rows


def _row_summary(tiles: list[np.ndarray]) -> dict:
    sigs = [tile.tobytes() for tile in tiles]
    counts = Counter(sigs)
    assert sorted(counts.values()) == [1, 3], "each row must contain one outlier and three copies"

    majority_sig = next(sig for sig, count in counts.items() if count == 3)
    outlier_idx = next(i for i, sig in enumerate(sigs) if counts[sig] == 1)
    majority = next(tile for tile, sig in zip(tiles, sigs) if sig == majority_sig).copy()
    outlier = tiles[outlier_idx].copy()

    for tile in tiles:
        assert set(map(int, np.unique(tile))).issubset({BG, FG}), "tiles must contain only black and blue"
        blue_count = int(np.sum(tile == FG))
        assert 3 <= blue_count <= 6, "tile sprite must contain 3-6 blue pixels"
        assert _is_connected(tile == FG), "tile sprite must be 4-connected"

    diff = outlier != majority
    diff_count = int(diff.sum())
    assert 1 <= diff_count <= 3, "outlier must differ in 1-3 cells"

    missing = int(np.sum((majority == FG) & (outlier == BG)))
    extra = int(np.sum((majority == BG) & (outlier == FG)))
    assert missing + extra == diff_count
    mode = "mixed" if missing and extra else ("missing" if missing else "extra")

    erased_tiles = []
    for tile in tiles:
        erased = tile.copy()
        erased[diff] = BG
        erased_tiles.append(erased)
    assert all(np.array_equal(erased_tiles[0], tile) for tile in erased_tiles[1:]), "mask should unify the row"

    return {
        "majority": majority,
        "outlier": outlier,
        "outlier_idx": outlier_idx,
        "diff": diff,
        "diff_count": diff_count,
        "mode": mode,
        "result": erased_tiles[0],
    }


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected 2D grid"
    row_lines, col_lines = _find_lattice(g)
    tile_rows = _extract_tiles(g, row_lines, col_lines)
    row_info = [_row_summary(tiles) for tiles in tile_rows]
    return {"grid": g, "row_lines": row_lines, "col_lines": col_lines, "rows": row_info}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each row's odd tile as an erase-mask instruction for the whole tile row.

    Input structure:
    - A grey separator lattice divides the grid into 3 rows by 4 columns of 3x3 tile interiors.
    - Each tile interior contains a black/blue 4-connected sprite with 3-6 blue pixels.
    - Within each tile row, three interiors are identical and one interior is an outlier.

    Rule:
    - For each tile row, identify the majority 3x3 sprite and the single outlier sprite.
    - Compare the outlier and majority cell-by-cell; every differing local coordinate becomes that row's erase mask.
    - Apply that local erase mask to every tile in the same row by setting those coordinates to black, while leaving all other cells and all grey separators unchanged.

    Key insight:
    - The outlier tile is not the answer to extract; its pixel differences against the row majority encode local coordinates that must be broadcast as an erase mask across the whole row.
    """
    info = _analyze(grid)
    g = info["grid"]
    out = g.copy()

    for row_idx, row in enumerate(info["rows"]):
        diff = row["diff"]
        r0 = info["row_lines"][row_idx] + 1
        r1 = info["row_lines"][row_idx + 1]
        for col_idx in range(COL_COUNT):
            c0 = info["col_lines"][col_idx] + 1
            c1 = info["col_lines"][col_idx + 1]
            tile = out[r0:r1, c0:c1]
            tile[diff] = BG

    return out


def _sample_connected_mask(rng: random.Random, min_cells: int = 3, max_cells: int = 6) -> np.ndarray:
    for _ in range(200):
        count = rng.randint(min_cells, max_cells)
        cells = {(rng.randrange(TILE), rng.randrange(TILE))}
        while len(cells) < count:
            r, c = rng.choice(tuple(cells))
            candidates = [(nr, nc) for nr, nc in _neighbors4(r, c) if 0 <= nr < TILE and 0 <= nc < TILE]
            rng.shuffle(candidates)
            for nr, nc in candidates:
                cells.add((nr, nc))
                if len(cells) == count:
                    break
        mask = np.zeros((TILE, TILE), dtype=int)
        for r, c in cells:
            mask[r, c] = FG
        if _is_connected(mask == FG):
            return mask
    raise RuntimeError("failed to sample connected 3x3 sprite")


def _adjacent_zeroes(mask: np.ndarray) -> list[tuple[int, int]]:
    zeros = []
    for r in range(TILE):
        for c in range(TILE):
            if mask[r, c] != BG:
                continue
            if any(0 <= nr < TILE and 0 <= nc < TILE and mask[nr, nc] == FG for nr, nc in _neighbors4(r, c)):
                zeros.append((r, c))
    return zeros


def _variant_missing(majority: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    ones = [(int(r), int(c)) for r, c in np.argwhere(majority == FG)]
    candidates = ones[:]
    rng.shuffle(candidates)
    for remove_count in rng.sample([1, 2, 3], k=3):
        if remove_count > len(ones) - 3:
            continue
        for _ in range(24):
            removed = rng.sample(candidates, remove_count)
            variant = majority.copy()
            for r, c in removed:
                variant[r, c] = BG
            if _is_connected(variant == FG):
                diff = variant != majority
                return variant, diff
    return None


def _variant_extra(majority: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    if int(np.sum(majority == FG)) >= 6:
        return None
    for add_count in rng.sample([1, 2, 3], k=3):
        if int(np.sum(majority == FG)) + add_count > 6:
            continue
        for _ in range(30):
            variant = majority.copy()
            frontier = _adjacent_zeroes(variant)
            if len(frontier) < add_count:
                continue
            added: list[tuple[int, int]] = []
            for _step in range(add_count):
                frontier = _adjacent_zeroes(variant)
                frontier = [cell for cell in frontier if cell not in added]
                if not frontier:
                    break
                cell = rng.choice(frontier)
                variant[cell] = FG
                added.append(cell)
            if len(added) == add_count and _is_connected(variant == FG):
                diff = variant != majority
                return variant, diff
    return None


def _variant_mixed(majority: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    ones = [(int(r), int(c)) for r, c in np.argwhere(majority == FG)]
    if len(ones) < 4:
        return None
    options = [(1, 1), (1, 2), (2, 1)]
    rng.shuffle(options)
    for remove_count, add_count in options:
        if len(ones) - remove_count < 3 or len(ones) - remove_count + add_count > 6:
            continue
        for _ in range(40):
            removed = rng.sample(ones, remove_count)
            variant = majority.copy()
            for r, c in removed:
                variant[r, c] = BG
            if not _is_connected(variant == FG):
                continue
            added: list[tuple[int, int]] = []
            for _step in range(add_count):
                frontier = _adjacent_zeroes(variant)
                frontier = [cell for cell in frontier if cell not in added]
                if not frontier:
                    break
                cell = rng.choice(frontier)
                variant[cell] = FG
                added.append(cell)
            if len(added) != add_count:
                continue
            if _is_connected(variant == FG):
                diff = variant != majority
                if 1 <= int(diff.sum()) <= 3:
                    return variant, diff
    return None


def _make_row(mode: str, rng: random.Random, forbidden_results: set[bytes]) -> dict | None:
    for _ in range(200):
        if mode == "missing":
            majority = _sample_connected_mask(rng, min_cells=4, max_cells=6)
            built = _variant_missing(majority, rng)
        elif mode == "extra":
            majority = _sample_connected_mask(rng, min_cells=3, max_cells=5)
            built = _variant_extra(majority, rng)
        else:
            majority = _sample_connected_mask(rng, min_cells=4, max_cells=5)
            built = _variant_mixed(majority, rng)
        if built is None:
            continue

        outlier, diff = built
        result = majority.copy()
        result[diff] = BG
        if int(np.sum(result == FG)) < 2:
            continue
        if not _is_connected(result == FG):
            continue
        if result.tobytes() in forbidden_results:
            continue

        return {
            "mode": mode,
            "majority": majority,
            "outlier": outlier,
            "diff": diff,
            "outlier_col": rng.randrange(COL_COUNT),
            "result": result,
        }
    return None


def _build_grid(rows: list[dict], rng: random.Random) -> np.ndarray:
    top = rng.randint(0, 2)
    bottom = rng.randint(0, 2)
    left = rng.randint(0, 2)
    right = rng.randint(0, 2)
    grid = np.full((top + LATTICE_H + bottom, left + LATTICE_W + right), BG, dtype=int)

    rr0 = top
    cc0 = left
    rr1 = rr0 + LATTICE_H - 1
    cc1 = cc0 + LATTICE_W - 1
    for r in range(rr0, rr1 + 1, TILE + 1):
        grid[r, cc0 : cc1 + 1] = SEP
    for c in range(cc0, cc1 + 1, TILE + 1):
        grid[rr0 : rr1 + 1, c] = SEP

    for row_idx, row in enumerate(rows):
        for col_idx in range(COL_COUNT):
            tile = row["outlier"] if col_idx == row["outlier_col"] else row["majority"]
            r0 = rr0 + 1 + row_idx * (TILE + 1)
            c0 = cc0 + 1 + col_idx * (TILE + 1)
            grid[r0 : r0 + TILE, c0 : c0 + TILE] = tile

    return grid


def _validate_generated_case(grid: np.ndarray) -> bool:
    try:
        info = _analyze(grid)
    except AssertionError:
        return False

    modes = [row["mode"] for row in info["rows"]]
    if sorted(modes) != ["extra", "missing", "mixed"]:
        return False

    result_sigs = {row["result"].tobytes() for row in info["rows"]}
    if len(result_sigs) != ROW_COUNT:
        return False

    diff_sigs = {row["diff"].astype(np.uint8).tobytes() for row in info["rows"]}
    if len(diff_sigs) != ROW_COUNT:
        return False

    if any(int(row["majority"].sum()) == int(row["result"].sum()) for row in info["rows"] if row["mode"] != "extra"):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    try:
        row_lines, col_lines = _find_lattice(out)
        out_tiles = _extract_tiles(out, row_lines, col_lines)
    except AssertionError:
        return False
    row_results = []
    for tiles in out_tiles:
        if not all(np.array_equal(tiles[0], tile) for tile in tiles[1:]):
            return False
        row_results.append(tiles[0].tobytes())
    if len(set(row_results)) != ROW_COUNT:
        return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        modes = ["missing", "mixed", "extra"]
        rng.shuffle(modes)

        rows = []
        result_sigs: set[bytes] = set()
        ok = True
        for mode in modes:
            row = _make_row(mode, rng, result_sigs)
            if row is None:
                ok = False
                break
            rows.append(row)
            result_sigs.add(row["result"].tobytes())
        if not ok:
            continue

        grid = _build_grid(rows, rng)
        if _validate_generated_case(grid):
            return grid

    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
