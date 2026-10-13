import numpy as np


TILE_SIZE = 3
TRANSFORM_COUNT = 8
INVERSE_TRANSFORM = (0, 3, 2, 1, 4, 5, 6, 7)
COLOR_POOL = np.arange(1, 10, dtype=int)
MAX_GENERATION_ATTEMPTS = 400


def _apply_transform(tile: np.ndarray, transform_id: int) -> np.ndarray:
    t = int(transform_id)
    if t == 0:
        return tile.copy()
    if t == 1:
        return np.rot90(tile, 1)
    if t == 2:
        return np.rot90(tile, 2)
    if t == 3:
        return np.rot90(tile, 3)
    if t == 4:
        return np.fliplr(tile)
    if t == 5:
        return np.flipud(tile)
    if t == 6:
        return tile.T.copy()
    if t == 7:
        return np.transpose(np.rot90(tile, 2)).copy()
    raise ValueError(f"unknown transform id: {transform_id}")


def _inverse_transform(tile: np.ndarray, transform_id: int) -> np.ndarray:
    return _apply_transform(tile, INVERSE_TRANSFORM[int(transform_id)])


LAWS = (
    {
        "name": "row_parity_lr",
        "min_rows": 3,
        "min_cols": 3,
        "transform": lambda r, c: 4 if (r % 2) == 1 else 0,
    },
    {
        "name": "col_parity_ud",
        "min_rows": 3,
        "min_cols": 3,
        "transform": lambda r, c: 5 if (c % 2) == 1 else 0,
    },
    {
        "name": "checker_rot180",
        "min_rows": 3,
        "min_cols": 3,
        "transform": lambda r, c: 2 if ((r + c) % 2) == 1 else 0,
    },
    {
        "name": "row_cycle_rot",
        "min_rows": 4,
        "min_cols": 3,
        "transform": lambda r, c: r % 4,
    },
    {
        "name": "col_cycle_rot",
        "min_rows": 3,
        "min_cols": 4,
        "transform": lambda r, c: c % 4,
    },
    {
        "name": "diag_cycle_rot",
        "min_rows": 3,
        "min_cols": 3,
        "transform": lambda r, c: (r + c) % 4,
    },
)


def _extract_tiles(grid: np.ndarray) -> tuple[list[list[np.ndarray]], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h % TILE_SIZE == 0 and w % TILE_SIZE == 0, "grid must be tiled by 3x3 blocks"
    rows = h // TILE_SIZE
    cols = w // TILE_SIZE
    tiles: list[list[np.ndarray]] = []
    for r in range(rows):
        row_tiles: list[np.ndarray] = []
        for c in range(cols):
            row_tiles.append(
                g[
                    r * TILE_SIZE : (r + 1) * TILE_SIZE,
                    c * TILE_SIZE : (c + 1) * TILE_SIZE,
                ].copy()
            )
        tiles.append(row_tiles)
    return tiles, rows, cols


def _assemble_grid(tiles: list[list[np.ndarray]]) -> np.ndarray:
    rows = len(tiles)
    cols = len(tiles[0])
    out = np.zeros((rows * TILE_SIZE, cols * TILE_SIZE), dtype=int)
    for r in range(rows):
        for c in range(cols):
            out[
                r * TILE_SIZE : (r + 1) * TILE_SIZE,
                c * TILE_SIZE : (c + 1) * TILE_SIZE,
            ] = tiles[r][c]
    return out


def _tile_key(tile: np.ndarray) -> bytes:
    return np.asarray(tile, dtype=int).tobytes()


def _fit_candidates(grid: np.ndarray) -> list[dict]:
    tiles, rows, cols = _extract_tiles(grid)
    total = rows * cols
    candidates: list[dict] = []

    for law in LAWS:
        counts: dict[bytes, int] = {}
        examples: dict[bytes, np.ndarray] = {}
        normalized_tiles: list[list[np.ndarray]] = []
        for r in range(rows):
            normalized_row: list[np.ndarray] = []
            for c in range(cols):
                normalized = _inverse_transform(tiles[r][c], law["transform"](r, c))
                normalized_row.append(normalized)
                key = _tile_key(normalized)
                counts[key] = counts.get(key, 0) + 1
                examples[key] = normalized
            normalized_tiles.append(normalized_row)

        majority_count = max(counts.values())
        if majority_count != total - 1:
            continue

        majority_keys = [key for key, count in counts.items() if count == majority_count]
        if len(majority_keys) != 1:
            continue

        base_tile = examples[majority_keys[0]]
        mismatch_positions: list[tuple[int, int]] = []
        for r in range(rows):
            for c in range(cols):
                if not np.array_equal(normalized_tiles[r][c], base_tile):
                    mismatch_positions.append((r, c))
        if len(mismatch_positions) != 1:
            continue

        bad_r, bad_c = mismatch_positions[0]
        expected_tile = _apply_transform(base_tile, law["transform"](bad_r, bad_c))
        if np.array_equal(expected_tile, tiles[bad_r][bad_c]):
            continue

        candidates.append(
            {
                "law_name": law["name"],
                "base_tile": base_tile,
                "bad_pos": (bad_r, bad_c),
                "expected_tile": expected_tile,
            }
        )

    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the hidden tile-layout law from the majority of a 3x3 tiling and output the corrected violating tile.

    Input structure:
    - The whole grid is partitioned into 3x3 tiles.
    - All but one tile come from a single unknown 3x3 base tile under a global row/column-dependent transform law.
    - The valid law may use flips or quarter-turn rotations, but it is consistent across the full tile array.

    Rule:
    - Split the input into 3x3 tiles.
    - For each plausible global transform law, undo the law at every tile position and look for the unique law under which exactly one tile disagrees while all others reduce to the same base tile.
    - Return the 3x3 tile that should appear at the violating position under that unique law.

    Key insight:
    - The bad tile cannot be judged in isolation, because it may itself be another legal-looking transform of the base tile.
    - Only the majority-consistent relationship between tile contents and tile coordinates reveals both the true law and the correct repair.
    """
    candidates = _fit_candidates(np.asarray(grid, dtype=int))
    assert len(candidates) == 1, "expected a unique hidden transform law with one violating tile"
    return candidates[0]["expected_tile"]


def _sample_base_tile(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        color_count = int(rng.integers(3, 7))
        colors = rng.choice(COLOR_POOL, size=color_count, replace=False)
        tile = rng.choice(colors, size=(TILE_SIZE, TILE_SIZE), replace=True)
        if len(np.unique(tile)) < 3:
            continue
        variants = [_apply_transform(tile, transform_id) for transform_id in range(TRANSFORM_COUNT)]
        if len({_tile_key(variant) for variant in variants}) != TRANSFORM_COUNT:
            continue
        return tile.astype(int)
    raise RuntimeError("failed to sample a sufficiently asymmetric base tile")


def _sample_wrong_tile(
    rng: np.random.Generator,
    base_tile: np.ndarray,
    expected_tile: np.ndarray,
    legal_tiles: list[np.ndarray],
) -> tuple[np.ndarray, str]:
    borrowed_tiles = [tile for tile in legal_tiles if not np.array_equal(tile, expected_tile)]
    all_transforms = [
        _apply_transform(base_tile, transform_id)
        for transform_id in range(TRANSFORM_COUNT)
        if not np.array_equal(_apply_transform(base_tile, transform_id), expected_tile)
    ]

    for _ in range(100):
        roll = float(rng.random())
        if borrowed_tiles and roll < 0.60:
            candidate = borrowed_tiles[int(rng.integers(len(borrowed_tiles)))]
            return candidate.copy(), "borrowed_visible"
        if all_transforms and roll < 0.90:
            candidate = all_transforms[int(rng.integers(len(all_transforms)))]
            return candidate.copy(), "borrowed_hidden"

        color_count = int(rng.integers(3, 7))
        colors = rng.choice(COLOR_POOL, size=color_count, replace=False)
        candidate = rng.choice(colors, size=(TILE_SIZE, TILE_SIZE), replace=True).astype(int)
        if np.array_equal(candidate, expected_tile):
            continue
        return candidate, "arbitrary"

    raise RuntimeError("failed to sample a wrong tile")


def _pick_dimensions(rng: np.random.Generator, law: dict) -> tuple[int, int]:
    rows = int(rng.integers(law["min_rows"], 6))
    cols = int(rng.integers(law["min_cols"], 6))
    return rows, cols


def _validate_generated_instance(
    grid: np.ndarray,
    law_name: str,
    bad_pos: tuple[int, int],
    expected_tile: np.ndarray,
) -> bool:
    candidates = _fit_candidates(grid)
    if len(candidates) != 1:
        return False
    candidate = candidates[0]
    if candidate["law_name"] != law_name:
        return False
    if candidate["bad_pos"] != bad_pos:
        return False
    if not np.array_equal(candidate["expected_tile"], expected_tile):
        return False
    if not np.array_equal(main(grid), expected_tile):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(MAX_GENERATION_ATTEMPTS):
        law = LAWS[int(rng.integers(len(LAWS)))]
        rows, cols = _pick_dimensions(rng, law)
        base_tile = _sample_base_tile(rng)

        legal_tiles: list[np.ndarray] = []
        tiles: list[list[np.ndarray]] = []
        for r in range(rows):
            row_tiles: list[np.ndarray] = []
            for c in range(cols):
                tile = _apply_transform(base_tile, law["transform"](r, c))
                row_tiles.append(tile)
                legal_tiles.append(tile)
            tiles.append(row_tiles)

        bad_r = int(rng.integers(rows))
        bad_c = int(rng.integers(cols))
        expected_tile = tiles[bad_r][bad_c].copy()
        wrong_tile, wrong_mode = _sample_wrong_tile(rng, base_tile, expected_tile, legal_tiles)
        if np.array_equal(wrong_tile, expected_tile):
            continue

        tiles[bad_r][bad_c] = wrong_tile
        grid = _assemble_grid(tiles)

        if not _validate_generated_instance(grid, law["name"], (bad_r, bad_c), expected_tile):
            continue

        # Ensure the visible violation is not accidentally self-explanatory.
        observed_bad_tile = grid[
            bad_r * TILE_SIZE : (bad_r + 1) * TILE_SIZE,
            bad_c * TILE_SIZE : (bad_c + 1) * TILE_SIZE,
        ]
        if np.array_equal(observed_bad_tile, expected_tile):
            continue

        # Most examples should force global reasoning by using another legal-looking transform.
        if wrong_mode != "arbitrary":
            normalized = _inverse_transform(observed_bad_tile, law["transform"](bad_r, bad_c))
            if np.array_equal(normalized, base_tile):
                continue

        return grid

    raise RuntimeError("failed to generate a valid tiling-law input")
