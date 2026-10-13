import numpy as np

from arc_utils import Color, get_bg
import random


BLOCK = 3


def main(grid: np.ndarray) -> np.ndarray:
    """
    Summarize each 3x3 input block as one output pixel, ignoring grey marks.
    Input structure
    - The grid dimensions are multiples of 3, forming a lattice of 3x3 blocks.
    - Background is the dominant color, and grey pixels are marker/noise cells.
    - Each real block contains one non-background, non-grey color in at least
      eight of its nine cells.
    Rule
    - Replace each 3x3 block by background if it contains only background and grey.
    - Otherwise replace it by the unique non-background, non-grey color in that block.
    Key insight
    - The large grid is an enlarged low-resolution picture; grey cells are defects
      within or between blocks and should not become output colors.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape
    assert h % BLOCK == 0 and w % BLOCK == 0, "grid must split into 3x3 blocks"

    out = np.full((h // BLOCK, w // BLOCK), bg, dtype=int)
    ignored = {bg, Color.GREY}

    for br in range(h // BLOCK):
        for bc in range(w // BLOCK):
            r0 = br * BLOCK
            c0 = bc * BLOCK
            block = g[r0 : r0 + BLOCK, c0 : c0 + BLOCK]
            colors = [int(c) for c in np.unique(block) if int(c) not in ignored]
            assert len(colors) <= 1, "each block has at most one real color"

            if colors:
                color = colors[0]
                assert np.count_nonzero(block == color) >= 8, "real blocks are full except grey damage"
                assert np.all((block == color) | (block == Color.GREY)), "real block damage must be grey"
                out[br, bc] = color
            else:
                assert np.all((block == bg) | (block == Color.GREY)), "empty blocks contain only bg/grey"

    return out

# --------------- generator (o5x) ---------------
# Canvas/lattice bounds.
BG_COLOR, GREY_COLOR, BLOCK_SIZE = 0, 5, 3
BLOCK_COUNT_MIN, BLOCK_COUNT_MAX = 3, 5
COLORED_BLOCK_MIN, COLORED_BLOCK_MAX = 2, 5
# Palette/defect constraints.
BLOCK_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
TOTAL_GREY_DEFECTS, SEEDED_DEFECT_BLOCKS = 6, 2
COLORED_GREY_LIMIT, BG_GREY_LIMIT = 1, 2
GEN_RETRY_LIMIT = 300

def _sample_layout():
    """Return lattice dimensions, block positions, and color assignments."""
    n_blocks = random.randint(BLOCK_COUNT_MIN, BLOCK_COUNT_MAX)
    m_blocks = random.randint(BLOCK_COUNT_MIN, BLOCK_COUNT_MAX)
    k_blocks = random.randint(COLORED_BLOCK_MIN, COLORED_BLOCK_MAX)
    block_positions = [(r, c) for r in range(n_blocks) for c in range(m_blocks)]
    colored_positions = random.sample(block_positions, k_blocks)
    assigned_colors = random.sample(BLOCK_COLORS, k_blocks)
    return n_blocks, m_blocks, block_positions, colored_positions, dict(zip(colored_positions, assigned_colors))

def _render_colored_blocks(n_blocks, m_blocks, color_by_pos):
    """Create input/output arrays with colored 3x3 blocks painted."""
    output = np.full((n_blocks, m_blocks), BG_COLOR, dtype=int)
    inp = np.full((n_blocks * BLOCK_SIZE, m_blocks * BLOCK_SIZE), BG_COLOR, dtype=int)

    for (br, bc), color in color_by_pos.items():
        output[br, bc] = color
        r0, c0 = br * BLOCK_SIZE, bc * BLOCK_SIZE
        inp[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = color
    return inp, output

def _sample_grey_counts(block_positions, colored_positions, color_by_pos):
    """Return per-block grey counts capped by colored/background limits."""
    grey_counts = {pos: 0 for pos in block_positions}
    grey_counts[random.choice(colored_positions)] = 1
    bg_positions = [pos for pos in block_positions if pos not in color_by_pos]
    grey_counts[random.choice(bg_positions)] = 1

    remaining_slots = []
    for pos in block_positions:
        limit = COLORED_GREY_LIMIT if pos in color_by_pos else BG_GREY_LIMIT
        remaining_slots.extend([pos] * (limit - grey_counts[pos]))

    for pos in random.sample(remaining_slots, TOTAL_GREY_DEFECTS - SEEDED_DEFECT_BLOCKS):
        grey_counts[pos] += 1
    return grey_counts

def _apply_grey_defects(inp, grey_counts):
    """Place grey defects within each selected 3x3 block."""
    for (br, bc), defect_count in grey_counts.items():
        if defect_count == 0:
            continue
        r0, c0 = br * BLOCK_SIZE, bc * BLOCK_SIZE
        for cell in random.sample(range(BLOCK_SIZE * BLOCK_SIZE), defect_count):
            inp[r0 + cell // BLOCK_SIZE, c0 + cell % BLOCK_SIZE] = GREY_COLOR

def _gen_attempt():
    n_blocks, m_blocks, block_positions, colored_positions, color_by_pos = _sample_layout()
    inp, output = _render_colored_blocks(n_blocks, m_blocks, color_by_pos)
    grey_counts = _sample_grey_counts(block_positions, colored_positions, color_by_pos)
    _apply_grey_defects(inp, grey_counts)

    if np.array_equal(main(inp), output):
        return inp
    return None

def generate_input():
    for _ in range(GEN_RETRY_LIMIT):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid 5614dbcf input")
