import numpy as np

from arc_utils import crop_bbox, find_components, get_bg, paste


def main(grid: np.ndarray) -> np.ndarray:
    """Extract the unique filled 3x3 foreground object that uses three colors.

    Input structure
    - The grid has a dominant background color, black/0 in the examples.
    - The foreground consists of separated filled 3x3 objects.
    - Exactly one object uses three colors; the other objects use two colors.

    Rule
    - Crop and return the only 3x3 foreground object whose cells contain three
      distinct non-background colors.

    Key insight
    - The selected output object is distinguished by color diversity, not by
      its position or by any specific color id.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    components = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert len(components) > 1, "expected multiple separated foreground objects"

    candidates = []
    for component in components:
        crop = crop_bbox(g, component["bbox"])
        assert crop.shape == (3, 3), "expected every foreground object to occupy a 3x3 bbox"
        assert int(component["area"]) == 9, "expected each foreground object to be a filled 3x3 block"
        assert np.all(crop != bg), "expected each foreground object crop to contain no background cells"

        color_count = len(set(map(int, crop.ravel())))
        if color_count == 3:
            candidates.append(crop)
        else:
            assert color_count == 2, "expected non-selected objects to use exactly two colors"

    assert len(candidates) == 1, "expected exactly one three-color foreground object"
    return candidates[0]

# --------------- generator (o5x) ---------------
# Canvas and block geometry.
GRID_SHAPE, BLOCK_SHAPE = (15, 15), (3, 3)
BLOCK_CELL_COUNT = BLOCK_SHAPE[0] * BLOCK_SHAPE[1]
# Palette and color count choices.
NON_BG_COLORS = np.arange(1, 10)
TWO_COLOR_SPLITS = ((3, 6), (4, 5))
COIN_FLIP_PROBABILITY = 0.5
COLOR_OVERLAP_PROBABILITY = 0.2
THREE_COLOR_CUT_COUNT = 2
THREE_COLOR_CUT_POSITIONS = np.arange(1, BLOCK_CELL_COUNT)
# Placement bands and retry budget.
ROW_START_MIN, ROW_START_STOP = 1, 12
MIN_ROW_SEPARATION = BLOCK_SHAPE[0] + 1
LEFT_COLUMN_STARTS, RIGHT_COLUMN_STARTS = np.arange(2, 4), np.arange(7, 9)
MAX_GENERATION_ATTEMPTS = 300


def _gen_block(colors, counts, rng):
    cells = np.repeat(np.array(colors, dtype=int), np.array(counts, dtype=int))
    rng.shuffle(cells)
    return cells.reshape(BLOCK_SHAPE)


def _gen_color_groups(rng):
    colors = rng.permutation(NON_BG_COLORS).tolist()
    two_groups = [colors[0:2], colors[2:4], colors[4:6]]
    three_group = colors[6:9]

    if rng.random() < COLOR_OVERLAP_PROBABILITY:
        groups = two_groups + [three_group]
        src_i, dst_i = rng.choice(len(groups), size=2, replace=False)
        src_color = int(rng.choice(groups[int(src_i)]))
        dst_group = groups[int(dst_i)]
        replaceable = [i for i, color in enumerate(dst_group) if color != src_color]
        dst_group[int(rng.choice(replaceable))] = src_color

    return two_groups, three_group


def _gen_two_color_counts(rng):
    counts = list(TWO_COLOR_SPLITS[0] if rng.random() < COIN_FLIP_PROBABILITY else TWO_COLOR_SPLITS[1])
    if rng.random() < COIN_FLIP_PROBABILITY:
        counts.reverse()
    return counts


def _gen_three_color_counts(rng):
    cuts = np.sort(rng.choice(THREE_COLOR_CUT_POSITIONS, size=THREE_COLOR_CUT_COUNT, replace=False))
    counts = [int(cuts[0]), int(cuts[1] - cuts[0]), int(BLOCK_CELL_COUNT - cuts[1])]
    rng.shuffle(counts)
    return counts


def _gen_band_rows(rng):
    first = int(rng.integers(ROW_START_MIN, ROW_START_STOP))
    choices = [r for r in range(ROW_START_MIN, ROW_START_STOP) if abs(r - first) >= MIN_ROW_SEPARATION]
    second = int(rng.choice(choices))
    rows = [first, second]
    rng.shuffle(rows)
    return rows


def _gen_positions(rng):
    left_cols = rng.choice(LEFT_COLUMN_STARTS, size=2, replace=True).astype(int).tolist()
    right_cols = rng.choice(RIGHT_COLUMN_STARTS, size=2, replace=True).astype(int).tolist()
    positions = [(r, c) for r, c in zip(_gen_band_rows(rng), left_cols)]
    positions += [(r, c) for r, c in zip(_gen_band_rows(rng), right_cols)]
    rng.shuffle(positions)
    return positions


def _gen_attempt():
    rng = np.random.default_rng()
    two_groups, three_group = _gen_color_groups(rng)

    target = _gen_block(three_group, _gen_three_color_counts(rng), rng)
    blocks = [target]
    for colors in two_groups:
        blocks.append(_gen_block(colors, _gen_two_color_counts(rng), rng))
    rng.shuffle(blocks)

    grid = np.zeros(GRID_SHAPE, dtype=int)
    for block, (r, c) in zip(blocks, _gen_positions(rng)):
        paste(grid, block, r, c, transparent=None)

    try:
        if np.array_equal(main(grid), target):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        generated = _gen_attempt()
        if generated is not None:
            return generated
    raise RuntimeError("failed to generate a valid 9a4bb226 input")
