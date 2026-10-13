import numpy as np


GRID_SIZE = 9
BLOCK_SIZE = 3
MACRO_SIZE = 3
BACKGROUND = 0
PALETTE = tuple(range(1, 10))
GENERATION_RETRIES = 500


def _shape_id(mask: np.ndarray) -> int:
    bits = 0
    for bit in mask.astype(int).ravel():
        bits = (bits << 1) | int(bit)
    return bits


def _mask_from_id(mask_id: int) -> np.ndarray:
    bits = [(mask_id >> shift) & 1 for shift in range(8, -1, -1)]
    return np.array(bits, dtype=bool).reshape(BLOCK_SIZE, BLOCK_SIZE)


ALL_MASK_IDS = tuple(range(1, 1 << 9))
MASKS_BY_COUNT: dict[int, tuple[int, ...]] = {
    count: tuple(mask_id for mask_id in ALL_MASK_IDS if mask_id.bit_count() == count)
    for count in range(1, 10)
}


def _macro_slice(index: int) -> slice:
    start = index * BLOCK_SIZE
    return slice(start, start + BLOCK_SIZE)


def _extract_macro_data(grid: np.ndarray) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray], dict[tuple[int, int], int]]:
    occupancy = np.zeros((MACRO_SIZE, MACRO_SIZE), dtype=bool)
    block_masks: dict[tuple[int, int], np.ndarray] = {}
    block_colors: dict[tuple[int, int], int] = {}
    for br in range(MACRO_SIZE):
        for bc in range(MACRO_SIZE):
            block = grid[_macro_slice(br), _macro_slice(bc)]
            fg_mask = block != BACKGROUND
            if not fg_mask.any():
                continue
            occupancy[br, bc] = True
            colors = np.unique(block[fg_mask])
            assert len(colors) == 1, "each non-empty macro block must be monochrome"
            block_masks[(br, bc)] = fg_mask
            block_colors[(br, bc)] = int(colors[0])
    return occupancy, block_masks, block_colors


def _render_repaired_grid(generator_mask: np.ndarray, block_colors: dict[tuple[int, int], int]) -> np.ndarray:
    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for (br, bc), color in block_colors.items():
        block = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=int)
        block[generator_mask] = int(color)
        out[_macro_slice(br), _macro_slice(bc)] = block
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Repair every non-empty macro block into the global occupancy-mask generator.

    Input structure:
    - The grid is a 9x9 black canvas partitioned into a 3x3 macro-grid of 3x3 blocks.
    - Some macro blocks contain monochrome foreground patterns and the remaining macro blocks are fully black.
    - Exactly one non-empty macro block has the same foreground-cell shape as the whole-grid 3x3 occupancy mask of non-empty macro blocks.

    Rule:
    - Compute the 3x3 occupancy mask whose cells mark which macro blocks are non-empty.
    - Find the unique non-empty macro block whose internal 3x3 foreground mask matches that occupancy mask exactly, ignoring color.
    - Replace every non-empty macro block by that generator shape while preserving the original color of that block; keep empty macro blocks black.

    Key insight:
    - The true template is identified by comparing one block's micro-pattern against the entire picture's macro occupancy pattern, not by comparing blocks to each other.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 9x9 grid"

    occupancy, block_masks, block_colors = _extract_macro_data(g)
    assert block_masks, "expected at least one non-empty macro block"

    generator_mask = occupancy
    matches = [pos for pos, mask in block_masks.items() if np.array_equal(mask, generator_mask)]
    assert len(matches) == 1, "expected a unique generator block"
    return _render_repaired_grid(generator_mask, block_colors)


def _sample_generator_mask() -> np.ndarray:
    for _ in range(200):
        count = int(np.random.randint(4, 7))
        mask_id = int(np.random.choice(np.array(MASKS_BY_COUNT[count], dtype=int)))
        mask = _mask_from_id(mask_id)
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if len(rows) < 2 or len(cols) < 2:
            continue
        if np.all(mask[1] == 0) and np.all(mask[:, 1] == 0):
            continue
        return mask
    raise RuntimeError("failed to sample a valid generator mask")


def _choose_variant_mask(generator_mask: np.ndarray, target_count: int, used_ids: set[int]) -> np.ndarray | None:
    true_count = int(generator_mask.sum())
    candidates: list[tuple[int, int, int]] = []
    for mask_id in MASKS_BY_COUNT[target_count]:
        if mask_id in used_ids:
            continue
        mask = _mask_from_id(mask_id)
        hamming = int(np.count_nonzero(mask != generator_mask))
        if target_count == true_count and hamming < 2:
            continue
        overlap = int(np.count_nonzero(mask & generator_mask))
        if overlap == 0 or overlap == true_count:
            continue
        candidates.append((hamming, -overlap, int(mask_id)))

    if not candidates:
        return None

    candidates.sort()
    shortlist = [mask_id for _, _, mask_id in candidates[: min(20, len(candidates))]]
    chosen_id = int(np.random.choice(np.array(shortlist, dtype=int)))
    used_ids.add(chosen_id)
    return _mask_from_id(chosen_id)


def _sample_variant_masks(generator_mask: np.ndarray) -> list[np.ndarray] | None:
    true_count = int(generator_mask.sum())
    needed = true_count - 1
    counts = [true_count, true_count - 1, true_count + 1]
    filler = [true_count, true_count, true_count - 1, true_count + 1, true_count - 2, true_count + 2]
    np.random.shuffle(filler)
    for count in filler:
        if len(counts) >= needed:
            break
        if 1 <= count <= 8:
            counts.append(int(count))
    if len(counts) != needed:
        return None

    np.random.shuffle(counts)
    used_ids = {_shape_id(generator_mask)}
    variants: list[np.ndarray] = []
    for count in counts:
        variant = _choose_variant_mask(generator_mask, int(count), used_ids)
        if variant is None:
            return None
        variants.append(variant)

    areas = [int(mask.sum()) for mask in variants]
    if not any(area == true_count for area in areas):
        return None
    if not any(area < true_count for area in areas):
        return None
    if not any(area > true_count for area in areas):
        return None
    return variants


def _build_input_grid(
    generator_mask: np.ndarray,
    generator_pos: tuple[int, int],
    block_shapes: dict[tuple[int, int], np.ndarray],
    block_colors: dict[tuple[int, int], int],
) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    active_positions = {tuple(map(int, pos)) for pos in np.argwhere(generator_mask)}
    assert active_positions == set(block_shapes)
    assert generator_pos in active_positions
    for pos, mask in block_shapes.items():
        br, bc = pos
        color = int(block_colors[pos])
        block = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=int)
        block[mask] = color
        grid[_macro_slice(br), _macro_slice(bc)] = block
    return grid


def _validate_instance(grid: np.ndarray, expected_generator_pos: tuple[int, int], expected_generator_mask: np.ndarray) -> bool:
    if grid.shape != (GRID_SIZE, GRID_SIZE):
        return False

    occupancy, block_masks, block_colors = _extract_macro_data(grid)
    if len(block_masks) != int(expected_generator_mask.sum()):
        return False
    if not np.array_equal(occupancy, expected_generator_mask):
        return False
    if len(set(block_colors.values())) != len(block_colors):
        return False

    matches = [pos for pos, mask in block_masks.items() if np.array_equal(mask, occupancy)]
    if matches != [expected_generator_pos]:
        return False

    true_count = int(occupancy.sum())
    corruptions = [mask for pos, mask in block_masks.items() if pos != expected_generator_pos]
    if len(corruptions) != true_count - 1:
        return False
    corruption_ids = [_shape_id(mask) for mask in corruptions]
    if len(corruption_ids) != len(set(corruption_ids)):
        return False

    areas = [int(mask.sum()) for mask in corruptions]
    if areas.count(true_count) < 1:
        return False
    if not any(area < true_count for area in areas):
        return False
    if not any(area > true_count for area in areas):
        return False

    expected = _render_repaired_grid(occupancy, block_colors)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    generator_mask = _sample_generator_mask()
    active_positions = [tuple(map(int, pos)) for pos in np.argwhere(generator_mask)]
    generator_pos = active_positions[int(np.random.randint(len(active_positions)))]
    variants = _sample_variant_masks(generator_mask)
    if variants is None:
        return None

    np.random.shuffle(active_positions)
    colors = np.random.choice(np.array(PALETTE, dtype=int), size=len(active_positions), replace=False)
    block_colors = {pos: int(color) for pos, color in zip(active_positions, colors)}

    remaining_positions = [pos for pos in active_positions if pos != generator_pos]
    np.random.shuffle(remaining_positions)
    np.random.shuffle(variants)
    block_shapes = {generator_pos: generator_mask.copy()}
    for pos, variant in zip(remaining_positions, variants):
        block_shapes[pos] = variant.copy()

    grid = _build_input_grid(generator_mask, generator_pos, block_shapes, block_colors)
    if not _validate_instance(grid, generator_pos, generator_mask):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
