from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()
Color = _ARC_UTILS.Color
paste = _ARC_UTILS.paste


MASK_SIZE = 3
SPRITE_SIZE = 3
OUTPUT_SIZE = 9
PANEL_ACTIVE = Color.MAROON
SEPARATOR = Color.BLUE
BACKGROUND = Color.BLACK
SPRITE_COLORS = [Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL]
MAX_GENERATION_ATTEMPTS = 400


def _extract_layout(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_coords = np.argwhere(g == SEPARATOR)
    assert sep_coords.shape == (MASK_SIZE, 2), "expected one 3-cell blue separator bar"
    sep_col = int(sep_coords[0, 1])
    assert np.all(sep_coords[:, 1] == sep_col), "separator must be vertical"
    sep_rows = sorted(int(r) for r in sep_coords[:, 0])
    assert sep_rows == list(range(sep_rows[0], sep_rows[0] + MASK_SIZE)), "separator must be contiguous"
    top = sep_rows[0]

    assert sep_col >= MASK_SIZE and sep_col + MASK_SIZE < g.shape[1], "separator must split two 3x3 panels"
    left = g[top : top + MASK_SIZE, sep_col - MASK_SIZE : sep_col]
    right = g[top : top + MASK_SIZE, sep_col + 1 : sep_col + 1 + MASK_SIZE]
    assert left.shape == (MASK_SIZE, MASK_SIZE) and right.shape == (MASK_SIZE, MASK_SIZE)
    assert set(map(int, np.unique(left))) <= {BACKGROUND, PANEL_ACTIVE}, "left panel must be binary"
    assert set(map(int, np.unique(right))) <= {BACKGROUND, PANEL_ACTIVE}, "right panel must be binary"

    spacer_row = top + MASK_SIZE
    assert spacer_row < g.shape[0], "missing spacer row"
    assert np.all(g[spacer_row, :] == BACKGROUND), "expected a black spacer row under the masks"

    sprite_mask = ~np.isin(g, [BACKGROUND, SEPARATOR, PANEL_ACTIVE])
    sprite_coords = np.argwhere(sprite_mask)
    assert sprite_coords.size > 0, "expected one colored 3x3 sprite"
    sr0 = int(sprite_coords[:, 0].min())
    sr1 = int(sprite_coords[:, 0].max())
    sc0 = int(sprite_coords[:, 1].min())
    sc1 = int(sprite_coords[:, 1].max())
    assert (sr0, sr1, sc0, sc1) == (top + MASK_SIZE + 1, top + MASK_SIZE + 3, sc0, sc0 + 2), (
        "sprite must be a 3x3 block immediately below the spacer row"
    )
    sprite = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    assert sprite.shape == (SPRITE_SIZE, SPRITE_SIZE), "sprite must be 3x3"
    assert np.all(sprite != BACKGROUND), "sprite cannot contain black"
    assert np.all(sprite != SEPARATOR) and np.all(sprite != PANEL_ACTIVE), "sprite colors must be distinct from control colors"
    assert len(np.unique(sprite)) >= 2, "sprite must use at least two visible colors"
    assert np.array_equal(sprite_mask[sr0 : sr1 + 1, sc0 : sc1 + 1], np.ones((3, 3), dtype=bool)), (
        "sprite must fully occupy its 3x3 bbox"
    )

    allowed = np.zeros_like(g, dtype=bool)
    allowed[top : top + MASK_SIZE, sep_col - MASK_SIZE : sep_col] = True
    allowed[top : top + MASK_SIZE, sep_col] = True
    allowed[top : top + MASK_SIZE, sep_col + 1 : sep_col + 1 + MASK_SIZE] = True
    allowed[sr0 : sr1 + 1, sc0 : sc1 + 1] = True
    assert np.all((g == BACKGROUND) | allowed), "unexpected non-background pixels outside the designed layout"
    return left.copy(), right.copy(), sprite.copy()


def _render_output(nor_mask: np.ndarray, sprite: np.ndarray) -> np.ndarray:
    out = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=int)
    for r, c in np.argwhere(nor_mask):
        paste(out, sprite, int(r) * SPRITE_SIZE, int(c) * SPRITE_SIZE)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use the NOR of two 3x3 binary panels as a placement mask for copying a 3x3 sprite into a 3x3 output array of slots.

    Input structure:
    - The upper part contains two 3x3 binary mask panels made of black and maroon cells, separated by one vertical blue column.
    - A full black spacer row sits directly below the masks.
    - A fully filled 3x3 colored sprite sits below that spacer row, using colors other than black, blue, and maroon.

    Rule:
    - Read the left and right 3x3 mask panels on the two sides of the blue separator.
    - Compute their pixelwise NOR: a position is active exactly when both masks are black there.
    - Create a 9x9 black output partitioned into nine 3x3 slots, and copy the entire sprite into every slot whose corresponding NOR cell is active.

    Key insight:
    - The logical NOR is not the final picture; it is a scheduling mask that decides where the colored sprite gets stamped in the larger array.
    """
    left, right, sprite = _extract_layout(grid)
    nor_mask = (left == BACKGROUND) & (right == BACKGROUND)
    return _render_output(nor_mask, sprite)


def _sample_masks() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    total = MASK_SIZE * MASK_SIZE
    active_count = random.randint(2, 6)
    inactive_count = total - active_count
    if inactive_count < 3:
        return None

    cells = list(range(total))
    active_cells = set(random.sample(cells, active_count))
    inactive_cells = [idx for idx in cells if idx not in active_cells]

    labels = [1, 2, 3]
    labels.extend(random.choice([1, 2, 3]) for _ in range(inactive_count - 3))
    random.shuffle(labels)

    state = np.zeros((MASK_SIZE, MASK_SIZE), dtype=int)
    for idx in active_cells:
        r, c = divmod(idx, MASK_SIZE)
        state[r, c] = 0
    for idx, label in zip(inactive_cells, labels):
        r, c = divmod(idx, MASK_SIZE)
        state[r, c] = label

    left = np.zeros((MASK_SIZE, MASK_SIZE), dtype=int)
    right = np.zeros((MASK_SIZE, MASK_SIZE), dtype=int)
    left[state == 1] = PANEL_ACTIVE
    right[state == 2] = PANEL_ACTIVE
    left[state == 3] = PANEL_ACTIVE
    right[state == 3] = PANEL_ACTIVE

    nor_mask = state == 0
    if np.unique(state).size != 4:
        return None
    if len({tuple(row) for row in nor_mask.astype(int)}) == 1 and len({tuple(col) for col in nor_mask.T.astype(int)}) == 1:
        return None
    return left, right, nor_mask


def _sample_sprite() -> np.ndarray:
    color_count = random.choice([2, 3])
    colors = random.sample(SPRITE_COLORS, color_count)
    sprite = np.array([random.choice(colors) for _ in range(SPRITE_SIZE * SPRITE_SIZE)], dtype=int).reshape(SPRITE_SIZE, SPRITE_SIZE)
    if len(np.unique(sprite)) != color_count:
        return _sample_sprite()
    if len({tuple(row) for row in sprite}) == 1:
        return _sample_sprite()
    if len({tuple(col) for col in sprite.T}) == 1:
        return _sample_sprite()
    return sprite


def _build_input(left: np.ndarray, right: np.ndarray, sprite: np.ndarray) -> np.ndarray:
    top_pad = random.randint(0, 2)
    left_pad = random.randint(0, 3)
    right_pad = random.randint(0, 3)
    bottom_pad = random.randint(0, 2)
    sprite_col = random.randint(left_pad, left_pad + 4)

    height = top_pad + MASK_SIZE + 1 + SPRITE_SIZE + bottom_pad
    width = max(left_pad + 7 + right_pad, sprite_col + SPRITE_SIZE + random.randint(0, 2))
    grid = np.zeros((height, width), dtype=int)

    row0 = top_pad
    sep_col = left_pad + MASK_SIZE
    grid[row0 : row0 + MASK_SIZE, sep_col] = SEPARATOR
    grid[row0 : row0 + MASK_SIZE, sep_col - MASK_SIZE : sep_col] = left
    grid[row0 : row0 + MASK_SIZE, sep_col + 1 : sep_col + 1 + MASK_SIZE] = right
    paste(grid, sprite, row0 + MASK_SIZE + 1, sprite_col)
    return grid


def _expected_output_from_input(grid: np.ndarray) -> np.ndarray:
    left, right, sprite = _extract_layout(grid)
    nor_mask = (left == BACKGROUND) & (right == BACKGROUND)
    return _render_output(nor_mask, sprite)


def _semantic_instance_ok(grid: np.ndarray) -> bool:
    try:
        left, right, sprite = _extract_layout(grid)
    except AssertionError:
        return False

    nor_mask = (left == BACKGROUND) & (right == BACKGROUND)
    active = int(nor_mask.sum())
    if not (2 <= active <= 6):
        return False

    l_only = np.any((left == PANEL_ACTIVE) & (right == BACKGROUND))
    r_only = np.any((left == BACKGROUND) & (right == PANEL_ACTIVE))
    both = np.any((left == PANEL_ACTIVE) & (right == PANEL_ACTIVE))
    if not (l_only and r_only and both):
        return False

    if len(np.unique(sprite)) not in (2, 3):
        return False

    expected = _render_output(nor_mask, sprite)
    try:
        solved = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False

    slot_colors = []
    for r in range(MASK_SIZE):
        for c in range(MASK_SIZE):
            block = solved[r * 3 : (r + 1) * 3, c * 3 : (c + 1) * 3]
            slot_colors.append(int(block.sum() > 0))
            if nor_mask[r, c]:
                if not np.array_equal(block, sprite):
                    return False
            else:
                if np.any(block):
                    return False
    return 0 in slot_colors and 1 in slot_colors


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_masks()
    if sampled is None:
        return None
    left, right, _nor_mask = sampled
    sprite = _sample_sprite()
    grid = _build_input(left, right, sprite)
    if not _semantic_instance_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid NOR stamp puzzle input")
