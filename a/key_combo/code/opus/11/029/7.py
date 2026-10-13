import numpy as np


BG = 0
TILE_H = 3
TILE_W = 5
TILE_COUNT = 5
INPUT_SHAPE = (TILE_H * TILE_COUNT, TILE_W)
OUTPUT_SHAPE = (TILE_COUNT, TILE_COUNT)
COLOR_POOL = np.array([1, 2, 3, 4, 5], dtype=int)
MAX_GENERATION_ATTEMPTS = 400
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
_RNG = np.random.default_rng()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    if box is None:
        raise ValueError("empty mask")
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.tobytes()


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
        for dr, dc in DIRS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                pt = (int(nr), int(nc))
                if pt not in seen:
                    seen.add(pt)
                    stack.append(pt)
    return len(seen) == int(coords.shape[0])


def _symmetry_flags(mask: np.ndarray) -> dict[str, bool]:
    flags = {
        "h": np.array_equal(mask, np.flipud(mask)),
        "v": np.array_equal(mask, np.fliplr(mask)),
        "d": False,
        "a": False,
        "r90": False,
        "r180": np.array_equal(mask, np.rot90(mask, 2)),
    }
    if mask.shape[0] == mask.shape[1]:
        flags["d"] = np.array_equal(mask, mask.T)
        flags["a"] = np.array_equal(mask, np.flipud(np.fliplr(mask)).T)
        flags["r90"] = np.array_equal(mask, np.rot90(mask, 1))
    return flags


def _symmetry_level(mask: np.ndarray) -> tuple[int, tuple[str, ...], dict[str, bool]]:
    flags = _symmetry_flags(mask)
    reflections = tuple(axis for axis in ("h", "v", "d", "a") if flags[axis])
    reflection_count = len(reflections)

    if reflection_count == 4 and flags["r90"]:
        return 4, reflections, flags
    if flags["r90"] and reflection_count < 4:
        return 3, reflections, flags
    if flags["r180"] and reflection_count == 0:
        return 3, reflections, flags
    if reflection_count == 2:
        return 2, reflections, flags
    if reflection_count == 1:
        return 1, reflections, flags
    if reflection_count == 0:
        return 0, reflections, flags
    raise ValueError(f"unsupported symmetry combination: {flags}")


def _extract_tile_info(tile: np.ndarray) -> dict:
    colors = sorted(set(map(int, tile.flatten())) - {BG})
    assert len(colors) == 1, f"expected exactly one foreground color per tile, got {colors}"
    color = colors[0]
    mask = tile == color
    area = int(mask.sum())
    assert 3 <= area <= 6, f"unexpected area {area}"
    crop = _crop_mask(mask)
    level, reflections, flags = _symmetry_level(crop)
    return {
        "color": int(color),
        "mask": mask,
        "crop": crop,
        "level": int(level),
        "reflections": reflections,
        "flags": flags,
    }


def _build_output(colors_by_level: dict[int, int]) -> np.ndarray:
    out = np.zeros(OUTPUT_SHAPE, dtype=int)
    for level in range(TILE_COUNT):
        out[:, level] = int(colors_by_level[level])
    return out


def _enumerate_candidate_masks() -> tuple[dict[int, list[np.ndarray]], dict[str, list[np.ndarray]], dict[tuple[str, str], list[np.ndarray]]]:
    by_level: dict[int, list[np.ndarray]] = {level: [] for level in range(TILE_COUNT)}
    level1_by_axis: dict[str, list[np.ndarray]] = {axis: [] for axis in ("h", "v", "d", "a")}
    level2_by_pair: dict[tuple[str, str], list[np.ndarray]] = {
        ("h", "v"): [],
        ("d", "a"): [],
    }
    seen: set[tuple[tuple[int, int], bytes]] = set()

    for bits in range(1, 1 << (TILE_H * TILE_W)):
        full = np.array([(bits >> i) & 1 for i in range(TILE_H * TILE_W)], dtype=bool).reshape(TILE_H, TILE_W)
        area = int(full.sum())
        if not (3 <= area <= 6):
            continue
        if not _is_connected(full):
            continue

        crop = _crop_mask(full)
        key = _mask_key(crop)
        if key in seen:
            continue
        seen.add(key)

        try:
            level, reflections, _ = _symmetry_level(crop)
        except ValueError:
            continue

        by_level[level].append(crop)
        if level == 1:
            level1_by_axis[reflections[0]].append(crop)
        if level == 2:
            level2_by_pair[reflections].append(crop)

    for level, pool in by_level.items():
        assert pool, f"no candidates for level {level}"
    assert all(level1_by_axis[axis] for axis in level1_by_axis)
    assert all(level2_by_pair[pair] for pair in level2_by_pair)
    return by_level, level1_by_axis, level2_by_pair


_CANDIDATES_BY_LEVEL, _LEVEL1_BY_AXIS, _LEVEL2_BY_PAIR = _enumerate_candidate_masks()


def _sample_crop_for_level(level: int) -> np.ndarray:
    if level == 1:
        axis = tuple(_LEVEL1_BY_AXIS.keys())[int(_RNG.integers(len(_LEVEL1_BY_AXIS)))]
        pool = _LEVEL1_BY_AXIS[axis]
    elif level == 2:
        pair = tuple(_LEVEL2_BY_PAIR.keys())[int(_RNG.integers(len(_LEVEL2_BY_PAIR)))]
        pool = _LEVEL2_BY_PAIR[pair]
    else:
        pool = _CANDIDATES_BY_LEVEL[level]
    return pool[int(_RNG.integers(len(pool)))].copy()


def _embed_crop(crop: np.ndarray, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = map(int, crop.shape)
    r0 = int(_RNG.integers(TILE_H - h + 1))
    c0 = int(_RNG.integers(TILE_W - w + 1))
    tile = np.zeros((TILE_H, TILE_W), dtype=int)
    tile[r0 : r0 + h, c0 : c0 + w][crop] = int(color)
    return tile, (r0, c0)


def _sample_level_order() -> np.ndarray:
    while True:
        order = _RNG.permutation(TILE_COUNT)
        if not np.array_equal(order, np.arange(TILE_COUNT)):
            return order


def _validate_generated_sample(grid: np.ndarray) -> bool:
    if tuple(grid.shape) != INPUT_SHAPE:
        return False

    levels = []
    colors_by_level: dict[int, int] = {}
    crop_needed = False
    for idx in range(TILE_COUNT):
        tile = grid[idx * TILE_H : (idx + 1) * TILE_H, :]
        info = _extract_tile_info(tile)
        level = info["level"]
        color = info["color"]
        if level in colors_by_level:
            return False
        colors_by_level[level] = color
        levels.append(level)

        try:
            full_level, _, _ = _symmetry_level(info["mask"])
        except ValueError:
            full_level = None
        if full_level != level:
            crop_needed = True

    if sorted(levels) != list(range(TILE_COUNT)):
        return False
    if tuple(levels) == tuple(range(TILE_COUNT)):
        return False
    if len(set(colors_by_level.values())) != TILE_COUNT:
        return False
    if not crop_needed:
        return False

    expected = _build_output(colors_by_level)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    if actual.shape != OUTPUT_SHAPE or len(set(map(int, np.unique(actual)))) != TILE_COUNT:
        return False
    for col in range(TILE_COUNT):
        if len(set(map(int, actual[:, col]))) != 1:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop each stacked monochrome tile to its foreground pattern, rank the patterns by symmetry class, and emit their colors as vertical stripes in rank order.

    Input structure:
    - The input is a 15x5 black-background grid made of five 3x5 tiles stacked vertically with no gaps.
    - Each tile contains one connected foreground pattern in a single non-black color, with 3-6 colored cells.
    - The five tile patterns represent five distinct symmetry classes once their black padding is cropped away.

    Rule:
    - Split the grid into five 3x5 tiles.
    - In each tile, isolate the single foreground color, crop the pattern to its bounding box, and classify its symmetry rank:
      0 = no symmetry, 1 = exactly one reflection axis, 2 = exactly two reflection axes,
      3 = pure rotational symmetry without reflections, 4 = full dihedral symmetry.
    - Sort tiles by increasing symmetry rank, breaking ties by original top-to-bottom order.
    - Return a 5x5 output whose i-th column is filled entirely with the foreground color of the i-th ranked tile.

    Key insight:
    - The black padding around each tile is irrelevant; only the cropped foreground pattern reveals the hidden symmetry rank that determines the stripe order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert tuple(g.shape) == INPUT_SHAPE, f"expected shape {INPUT_SHAPE}, got {g.shape}"

    ranked = []
    levels = []
    for idx in range(TILE_COUNT):
        tile = g[idx * TILE_H : (idx + 1) * TILE_H, :]
        info = _extract_tile_info(tile)
        ranked.append((info["level"], idx, info["color"]))
        levels.append(info["level"])

    assert sorted(levels) == list(range(TILE_COUNT)), f"expected exactly one tile per level, got {levels}"

    ranked.sort(key=lambda item: (item[0], item[1]))
    out = np.zeros(OUTPUT_SHAPE, dtype=int)
    for col, (_, _, color) in enumerate(ranked):
        out[:, col] = int(color)
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        colors_by_level = {level: int(color) for level, color in enumerate(_RNG.permutation(COLOR_POOL))}
        input_order = _sample_level_order()

        tiles = []
        for level in input_order:
            crop = _sample_crop_for_level(int(level))
            tile, _ = _embed_crop(crop, colors_by_level[int(level)])
            tiles.append(tile)

        grid = np.vstack(tiles).astype(int)
        if _validate_generated_sample(grid):
            return grid

    raise RuntimeError("failed to generate a valid input for puzzle 7")
