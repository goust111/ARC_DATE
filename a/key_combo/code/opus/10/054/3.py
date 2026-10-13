import numpy as np


BLOCK = 3
BG = 0
SEPARATOR = 2
OUTPUT_FG = 3
GREY = 5
FG_PALETTE = (1, 4, 6, 7, 8, 9)
GEN_RETRY_LIMIT = 400

REL_UPPER_ONLY = "upper_only"
REL_LOWER_ONLY = "lower_only"
REL_BOTH_BG = "both_bg"
REL_BOTH_FG_SAME = "both_fg_same"
REL_BOTH_FG_DIFF = "both_fg_diff"
RELATION_TYPES = (
    REL_UPPER_ONLY,
    REL_LOWER_ONLY,
    REL_BOTH_BG,
    REL_BOTH_FG_SAME,
    REL_BOTH_FG_DIFF,
)


def _find_separator_row(grid: np.ndarray) -> int:
    candidates = np.flatnonzero(np.all(grid == SEPARATOR, axis=1))
    assert candidates.size == 1, "expected exactly one full-width red separator row"
    sep = int(candidates[0])
    assert np.count_nonzero(grid == SEPARATOR) == grid.shape[1], "red must appear only in the separator"
    return sep


def _block_main_color(block: np.ndarray) -> int:
    non_grey = block[block != GREY]
    assert non_grey.size >= 8, "each 3x3 block must keep at least eight non-grey cells"
    colors = np.unique(non_grey)
    assert colors.size == 1, "each 3x3 block must have a single non-grey main color"
    return int(colors[0])


def _compress_region(region: np.ndarray) -> np.ndarray:
    h, w = map(int, region.shape)
    assert h % BLOCK == 0 and w % BLOCK == 0, "region must decompose into 3x3 blocks"
    out = np.zeros((h // BLOCK, w // BLOCK), dtype=int)
    for br in range(out.shape[0]):
        for bc in range(out.shape[1]):
            r0 = br * BLOCK
            c0 = bc * BLOCK
            out[br, bc] = _block_main_color(region[r0 : r0 + BLOCK, c0 : c0 + BLOCK])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Denoise 3x3 block lattices above and below a separator, compress them, and output their coarse foreground xor.

    Input structure:
    - The grid has one full-width red separator row whose color appears nowhere else.
    - The separator splits the grid into equal-size upper and lower regions.
    - Each region is a lattice of 3x3 blocks where every block has one main color and optionally one grey noise pixel.

    Rule:
    - Remove the separator row.
    - For each 3x3 block in each half, ignore grey and read the remaining majority color as that block's main color.
    - Compress each half to one coarse cell per 3x3 block.
    - Compare the two coarse grids cell by cell and output green exactly where one coarse cell is non-black and the other is black; otherwise output black.

    Key insight:
    - The xor happens after denoising and 3x3 compression, so different foreground colors still cancel and grey noise must not create coarse foreground cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    sep = _find_separator_row(g)
    assert 2 * sep + 1 == h, "separator must split the grid into equal-height halves"

    upper = g[:sep, :]
    lower = g[sep + 1 :, :]
    assert upper.shape == lower.shape, "upper and lower halves must align"
    assert upper.shape[0] % BLOCK == 0 and w % BLOCK == 0, "halves must be divisible into 3x3 blocks"

    upper_coarse = _compress_region(upper)
    lower_coarse = _compress_region(lower)
    return np.where((upper_coarse != BG) ^ (lower_coarse != BG), OUTPUT_FG, BG).astype(int)


def _sample_foreground_color(rng: np.random.Generator, avoid: tuple[int, ...] = ()) -> int:
    palette = [color for color in FG_PALETTE if color not in avoid]
    return int(rng.choice(palette))


def _sample_relation_grid(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    relations = [
        REL_UPPER_ONLY,
        REL_LOWER_ONLY,
        REL_BOTH_BG,
        REL_BOTH_FG_SAME,
        REL_BOTH_FG_DIFF,
    ]
    if rows * cols < len(relations):
        raise ValueError("coarse grid is too small to cover all required branches")

    probs = np.array([0.23, 0.23, 0.18, 0.18, 0.18], dtype=float)
    extra = rng.choice(RELATION_TYPES, size=rows * cols - len(relations), p=probs)
    relations.extend(map(str, extra))
    rng.shuffle(relations)
    return np.array(relations, dtype=object).reshape(rows, cols)


def _build_coarse_grids(relation_grid: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = relation_grid.shape
    upper = np.zeros((rows, cols), dtype=int)
    lower = np.zeros((rows, cols), dtype=int)

    for r in range(rows):
        for c in range(cols):
            relation = relation_grid[r, c]
            if relation == REL_UPPER_ONLY:
                upper[r, c] = _sample_foreground_color(rng)
            elif relation == REL_LOWER_ONLY:
                lower[r, c] = _sample_foreground_color(rng)
            elif relation == REL_BOTH_BG:
                pass
            elif relation == REL_BOTH_FG_SAME:
                color = _sample_foreground_color(rng)
                upper[r, c] = color
                lower[r, c] = color
            elif relation == REL_BOTH_FG_DIFF:
                color1 = _sample_foreground_color(rng)
                color2 = _sample_foreground_color(rng, avoid=(color1,))
                upper[r, c] = color1
                lower[r, c] = color2
            else:
                raise AssertionError(f"unknown relation: {relation}")

    assert np.any(upper != BG), "upper coarse grid must contain foreground"
    assert np.any(lower != BG), "lower coarse grid must contain foreground"
    return upper, lower


def _choose_coord(coords: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    idx = int(rng.integers(len(coords)))
    return tuple(int(v) for v in coords[idx])


def _build_noise_masks(
    relation_grid: np.ndarray,
    upper_coarse: np.ndarray,
    lower_coarse: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = relation_grid.shape
    upper_noise = rng.random((rows, cols)) < 0.35
    lower_noise = rng.random((rows, cols)) < 0.35

    both_bg_coords = np.argwhere(relation_grid == REL_BOTH_BG)
    bg_r, bg_c = _choose_coord(both_bg_coords, rng)
    if rng.random() < 0.5:
        upper_noise[bg_r, bg_c] = True
        lower_noise[bg_r, bg_c] = False
    else:
        upper_noise[bg_r, bg_c] = False
        lower_noise[bg_r, bg_c] = True

    fg_coords = np.argwhere((upper_coarse != BG) | (lower_coarse != BG))
    fg_r, fg_c = _choose_coord(fg_coords, rng)
    if upper_coarse[fg_r, fg_c] != BG and lower_coarse[fg_r, fg_c] != BG:
        if rng.random() < 0.5:
            upper_noise[fg_r, fg_c] = True
        else:
            lower_noise[fg_r, fg_c] = True
    elif upper_coarse[fg_r, fg_c] != BG:
        upper_noise[fg_r, fg_c] = True
    else:
        lower_noise[fg_r, fg_c] = True

    return upper_noise, lower_noise


def _render_region(coarse: np.ndarray, noise_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    rows, cols = coarse.shape
    region = np.zeros((rows * BLOCK, cols * BLOCK), dtype=int)
    for r in range(rows):
        for c in range(cols):
            block = np.full((BLOCK, BLOCK), int(coarse[r, c]), dtype=int)
            if noise_mask[r, c]:
                cell = int(rng.integers(BLOCK * BLOCK))
                block[cell // BLOCK, cell % BLOCK] = GREY
            r0 = r * BLOCK
            c0 = c * BLOCK
            region[r0 : r0 + BLOCK, c0 : c0 + BLOCK] = block
    return region


def _compress_by_any_nonblack(region: np.ndarray) -> np.ndarray:
    rows = region.shape[0] // BLOCK
    cols = region.shape[1] // BLOCK
    coarse = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            block = region[r * BLOCK : (r + 1) * BLOCK, c * BLOCK : (c + 1) * BLOCK]
            coarse[r, c] = np.any(block != BG)
    return coarse


def _validate_instance(
    grid: np.ndarray,
    upper_coarse: np.ndarray,
    lower_coarse: np.ndarray,
    relation_grid: np.ndarray,
) -> None:
    expected = np.where((upper_coarse != BG) ^ (lower_coarse != BG), OUTPUT_FG, BG).astype(int)
    actual = main(grid)
    assert np.array_equal(actual, expected), "main() must match the intended coarse xor semantics"
    assert np.any(expected == OUTPUT_FG), "output must contain at least one xor foreground cell"
    assert np.any(expected == BG), "output must contain at least one background cell"

    wrong_color_sensitive = np.where(upper_coarse != lower_coarse, OUTPUT_FG, BG).astype(int)
    assert np.any(relation_grid == REL_BOTH_FG_DIFF), "generator must include the color-insensitive xor counterexample"
    assert not np.array_equal(wrong_color_sensitive, expected), "different foreground colors should still cancel"

    sep = _find_separator_row(grid)
    upper_region = grid[:sep, :]
    lower_region = grid[sep + 1 :, :]
    wrong_presence = np.where(
        _compress_by_any_nonblack(upper_region) ^ _compress_by_any_nonblack(lower_region),
        OUTPUT_FG,
        BG,
    ).astype(int)
    assert not np.array_equal(wrong_presence, expected), "grey-noisy background blocks must not become coarse foreground"


def _gen_attempt(rng: np.random.Generator) -> np.ndarray:
    coarse_rows = int(rng.integers(3, 5))
    coarse_cols = int(rng.integers(3, 7))

    relation_grid = _sample_relation_grid(coarse_rows, coarse_cols, rng)
    upper_coarse, lower_coarse = _build_coarse_grids(relation_grid, rng)
    upper_noise, lower_noise = _build_noise_masks(relation_grid, upper_coarse, lower_coarse, rng)

    upper_region = _render_region(upper_coarse, upper_noise, rng)
    lower_region = _render_region(lower_coarse, lower_noise, rng)

    grid = np.zeros((upper_region.shape[0] + lower_region.shape[0] + 1, upper_region.shape[1]), dtype=int)
    grid[: upper_region.shape[0], :] = upper_region
    grid[upper_region.shape[0], :] = SEPARATOR
    grid[upper_region.shape[0] + 1 :, :] = lower_region

    _validate_instance(grid, upper_coarse, lower_coarse, relation_grid)
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRY_LIMIT):
        try:
            return _gen_attempt(rng)
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input for puzzle 3")
