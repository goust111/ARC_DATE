from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import colors_present, get_bg
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
        return int(values[np.argmax(counts)])


    def colors_present(grid: np.ndarray, *, exclude=None) -> list[int]:
        colors = set(map(int, np.unique(np.asarray(grid, dtype=int))))
        if exclude is not None:
            colors -= set(map(int, exclude))
        return sorted(colors)


GRID_SIZE = 9
BLOCK_SIZE = 3
THUMB_SIZE = 3
CELL_COUNT = THUMB_SIZE * THUMB_SIZE
MAX_RETRIES = 400
PALETTE = tuple(range(10))

MASKS_BY_COUNT: dict[int, tuple[int, ...]] = {
    count: tuple(mask for mask in range(1, 1 << CELL_COUNT) if mask.bit_count() == count)
    for count in range(1, CELL_COUNT + 1)
}


def _infer_colors(grid: np.ndarray) -> tuple[int, int]:
    colors = colors_present(grid)
    assert len(colors) == 2, "expected exactly two colors"
    counts = {color: int(np.sum(grid == color)) for color in colors}
    fill = max(counts, key=counts.get)
    marker = min(counts, key=counts.get)
    assert counts[fill] > counts[marker], "expected a unique dominant fill color"
    return int(fill), int(marker)


def _block(grid: np.ndarray, br: int, bc: int) -> np.ndarray:
    r0 = br * BLOCK_SIZE
    c0 = bc * BLOCK_SIZE
    return grid[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE]


def _compress(grid: np.ndarray, fill: int, marker: int) -> np.ndarray:
    code = np.full((THUMB_SIZE, THUMB_SIZE), int(fill), dtype=int)
    for br in range(THUMB_SIZE):
        for bc in range(THUMB_SIZE):
            if np.any(_block(grid, br, bc) == marker):
                code[br, bc] = int(marker)
    return code


def _matching_positions(grid: np.ndarray, code: np.ndarray) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    for br in range(THUMB_SIZE):
        for bc in range(THUMB_SIZE):
            if np.array_equal(_block(grid, br, bc), code):
                matches.append((br, bc))
    return matches


def _mask_to_block(mask: int, fill: int, marker: int) -> np.ndarray:
    block = np.full((THUMB_SIZE, THUMB_SIZE), int(fill), dtype=int)
    for idx in range(CELL_COUNT):
        if (mask >> idx) & 1:
            block[idx // THUMB_SIZE, idx % THUMB_SIZE] = int(marker)
    return block


def _assemble_grid(block_masks: list[int], fill: int, marker: int) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), int(fill), dtype=int)
    for idx, mask in enumerate(block_masks):
        br, bc = divmod(idx, THUMB_SIZE)
        _block(grid, br, bc)[:, :] = _mask_to_block(mask, fill, marker)
    return grid


def _count_marker_blocks(code_mask: int) -> int:
    return int(code_mask.bit_count())


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 9x9 grid"
    fill, marker = _infer_colors(g)
    code = _compress(g, fill, marker)
    matches = _matching_positions(g, code)
    return {
        "fill": fill,
        "marker": marker,
        "code": code,
        "matches": matches,
        "marker_block_count": int(np.sum(code == marker)),
        "marker_counts_by_block": {
            (br, bc): int(np.sum(_block(g, br, bc) == marker))
            for br in range(THUMB_SIZE)
            for bc in range(THUMB_SIZE)
        },
    }


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Crop the unique 3x3 block whose exact contents equal the whole-grid thumbnail code.

    Input structure:
    - The input is a 9x9 grid partitioned into nine 3x3 blocks.
    - Exactly two colors appear: one dominant fill color and one sparser marker color.
    - Every block is either all fill or contains one or more marker pixels on the fill color.

    Rule:
    - Compress the grid to a 3x3 code by writing the marker color for each block that contains any marker pixel, otherwise the fill color.
    - Find the unique 3x3 block whose cell-by-cell contents exactly equal that compressed 3x3 code.
    - Return that matching block unchanged.

    Key insight:
    - The target block is identified only after computing the coarse thumbnail of the entire grid.
    - The winning block is self-referential: it participates in the compression while also storing the resulting thumbnail inside itself.
    """
    g = np.asarray(input_grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 9x9 grid"

    fill, marker = _infer_colors(g)
    code = _compress(g, fill, marker)
    matches = _matching_positions(g, code)
    assert len(matches) == 1, "expected exactly one thumbnail-matching block"
    br, bc = matches[0]
    return _block(g, br, bc).copy()


def _sample_code_mask() -> int:
    marker_block_count = random.randint(2, 7)
    return int(random.choice(MASKS_BY_COUNT[marker_block_count]))


def _sample_marker_block_mask(code_mask: int, desired_count: int | None = None) -> int:
    if desired_count is None:
        desired_count = random.choice((1, 1, 2, 2, 3, 4, 5))
    desired_count = max(1, min(CELL_COUNT, int(desired_count)))
    candidates = [mask for mask in MASKS_BY_COUNT[desired_count] if mask != code_mask]
    if not candidates:
        candidates = [mask for mask in range(1, 1 << CELL_COUNT) if mask != code_mask]
    return int(random.choice(candidates))


def _build_instance() -> np.ndarray:
    fill, marker = random.sample(PALETTE, 2)
    code_mask = _sample_code_mask()
    target_idx = random.choice([idx for idx in range(CELL_COUNT) if (code_mask >> idx) & 1])

    block_masks = [0] * CELL_COUNT
    block_masks[target_idx] = code_mask

    marker_positions = [idx for idx in range(CELL_COUNT) if (code_mask >> idx) & 1 and idx != target_idx]
    if marker_positions:
        same_count_idx = random.choice(marker_positions)
        block_masks[same_count_idx] = _sample_marker_block_mask(code_mask, code_mask.bit_count())

    for idx in range(CELL_COUNT):
        if idx == target_idx:
            continue
        if (code_mask >> idx) & 1:
            if block_masks[idx] == 0:
                desired = random.choice((1, 2, 3, max(1, code_mask.bit_count() - 1)))
                block_masks[idx] = _sample_marker_block_mask(code_mask, desired)
        else:
            block_masks[idx] = 0

    grid = _assemble_grid(block_masks, fill, marker)
    total_markers = int(np.sum(grid == marker))
    assert total_markers < (GRID_SIZE * GRID_SIZE) // 2, "marker color must stay sparse"
    return grid


def _is_semantically_valid(grid: np.ndarray) -> bool:
    try:
        info = _analyze(grid)
    except AssertionError:
        return False

    if len(info["matches"]) != 1:
        return False
    if not (2 <= info["marker_block_count"] <= 7):
        return False

    winner = info["matches"][0]
    winner_marker_count = info["marker_counts_by_block"][winner]
    other_same_count = [
        pos
        for pos, count in info["marker_counts_by_block"].items()
        if pos != winner and count == winner_marker_count
    ]
    if not other_same_count:
        return False

    nonempty_blocks = sum(1 for count in info["marker_counts_by_block"].values() if count > 0)
    empty_blocks = CELL_COUNT - nonempty_blocks
    if nonempty_blocks < 2 or empty_blocks < 2:
        return False

    code = info["code"]
    if np.all(code == info["fill"]) or np.all(code == info["marker"]):
        return False

    winner_block = _block(np.asarray(grid, dtype=int), *winner)
    if not np.array_equal(winner_block, code):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _build_instance()
        if _is_semantically_valid(grid):
            return grid
    raise RuntimeError("failed to generate a valid thumbnail-in-thumbnail instance")
