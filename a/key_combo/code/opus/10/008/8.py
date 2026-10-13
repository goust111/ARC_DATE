from __future__ import annotations

import random
from typing import Dict, Iterable, List, Tuple

import numpy as np


BACKGROUND = 0
BLUE = 1
YELLOW = 4
MAGENTA = 6

CELL_SIZE = 5
LINE_WIDTH = 1
STRIDE = CELL_SIZE + LINE_WIDTH
GRID_CELLS = 3
GRID_SIZE = GRID_CELLS * CELL_SIZE + (GRID_CELLS - 1) * LINE_WIDTH
INSTRUCTION_CELL = (2, 2)
GEN_ATTEMPTS = 400

# Five visually distinct 2x2 placements inside a 5x5 cell.
BLOCK_TOP_LEFTS: Tuple[Tuple[int, int], ...] = (
    (0, 0),  # top-left
    (0, 2),  # top-right
    (1, 1),  # center
    (2, 0),  # bottom-left
    (2, 2),  # bottom-right
)


def _full_color_rows(grid: np.ndarray, color: int) -> List[int]:
    return [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]


def _full_color_cols(grid: np.ndarray, color: int) -> List[int]:
    return [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]


def _cell_starts_from_lines(lines: Iterable[int], extent: int) -> List[int]:
    starts: List[int] = []
    prev = -1
    for line in list(lines) + [extent]:
        starts.append(prev + 1)
        prev = int(line)
    return starts


def _parse_layout(grid: np.ndarray) -> Tuple[List[int], List[int]]:
    row_lines = _full_color_rows(grid, YELLOW)
    col_lines = _full_color_cols(grid, YELLOW)
    assert len(row_lines) == GRID_CELLS - 1
    assert len(col_lines) == GRID_CELLS - 1

    row_starts = _cell_starts_from_lines(row_lines, grid.shape[0])
    col_starts = _cell_starts_from_lines(col_lines, grid.shape[1])
    assert len(row_starts) == GRID_CELLS
    assert len(col_starts) == GRID_CELLS
    for start in row_starts:
        assert start + CELL_SIZE <= grid.shape[0]
    for start in col_starts:
        assert start + CELL_SIZE <= grid.shape[1]
    return row_starts, col_starts


def _cell_view(grid: np.ndarray, row_starts: List[int], col_starts: List[int], cr: int, cc: int) -> np.ndarray:
    r0 = row_starts[cr]
    c0 = col_starts[cc]
    return grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE]


def _extract_block(cell: np.ndarray) -> Tuple[int, int] | None:
    blue_coords = np.argwhere(cell == BLUE)
    if blue_coords.size == 0:
        return None
    assert len(blue_coords) == 4, "occupied cell must contain exactly one blue 2x2 block"
    r0 = int(blue_coords[:, 0].min())
    c0 = int(blue_coords[:, 1].min())
    assert r0 + 1 < CELL_SIZE and c0 + 1 < CELL_SIZE
    block = np.zeros_like(cell, dtype=bool)
    block[r0 : r0 + 2, c0 : c0 + 2] = True
    assert np.array_equal(block, cell == BLUE), "blue pixels must form one solid 2x2 block"
    return r0, c0


def _parse_blocks(grid: np.ndarray) -> Tuple[List[int], List[int], Dict[Tuple[int, int], Dict[str, int]], np.ndarray]:
    row_starts, col_starts = _parse_layout(grid)
    instruction = _cell_view(grid, row_starts, col_starts, *INSTRUCTION_CELL).copy()
    blocks: Dict[Tuple[int, int], Dict[str, int]] = {}

    for cr in range(GRID_CELLS):
        for cc in range(GRID_CELLS):
            cell = _cell_view(grid, row_starts, col_starts, cr, cc)
            if (cr, cc) == INSTRUCTION_CELL:
                assert np.all((cell == BACKGROUND) | (cell == MAGENTA))
                continue
            assert np.all((cell == BACKGROUND) | (cell == BLUE))
            block = _extract_block(cell)
            if block is None:
                continue
            top, left = block
            blocks[(cr, cc)] = {
                "top": top,
                "left": left,
                "anchor_r": top + 1,
                "anchor_c": left + 1,
            }

    return row_starts, col_starts, blocks, instruction


def _pattern_from_blocks(
    cell_key: Tuple[int, int],
    blocks: Dict[Tuple[int, int], Dict[str, int]],
    *,
    include_neighbors: bool,
) -> np.ndarray:
    info = blocks[cell_key]
    cr, cc = cell_key
    local = np.zeros((CELL_SIZE, CELL_SIZE), dtype=int)
    local[info["top"] : info["top"] + 2, info["left"] : info["left"] + 2] = BLUE

    horizontal_sources: List[Tuple[Tuple[int, int], int]] = [((cr, cc), info["anchor_r"])]
    vertical_sources: List[Tuple[Tuple[int, int], int]] = [((cr, cc), info["anchor_c"])]

    if include_neighbors:
        for other_key, other in blocks.items():
            if other_key == cell_key:
                continue
            orow, ocol = other_key
            if orow == cr:
                horizontal_sources.append((other_key, other["anchor_r"]))
            if ocol == cc:
                vertical_sources.append((other_key, other["anchor_c"]))

    for _, row in horizontal_sources:
        local[row, :] = BLUE
    for _, col in vertical_sources:
        local[:, col] = BLUE

    for h_key, row in horizontal_sources:
        for v_key, col in vertical_sources:
            if h_key != v_key:
                local[row, col] = BACKGROUND

    return local


def _processed_cells(blocks: Dict[Tuple[int, int], Dict[str, int]]) -> Dict[Tuple[int, int], np.ndarray]:
    return {key: _pattern_from_blocks(key, blocks, include_neighbors=True) for key in blocks}


def _own_only_cells(blocks: Dict[Tuple[int, int], Dict[str, int]]) -> Dict[Tuple[int, int], np.ndarray]:
    return {key: _pattern_from_blocks(key, blocks, include_neighbors=False) for key in blocks}


def _render_output_from_patterns(
    grid: np.ndarray,
    row_starts: List[int],
    col_starts: List[int],
    patterns: Dict[Tuple[int, int], np.ndarray],
    mask: np.ndarray,
) -> np.ndarray:
    out = grid.copy()
    for key, pattern in patterns.items():
        cr, cc = key
        cell = _cell_view(out, row_starts, col_starts, cr, cc)
        cell[:] = BACKGROUND
        if np.array_equal(pattern == BLUE, mask):
            cell[pattern == BLUE] = MAGENTA
        else:
            cell[pattern == BLUE] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect which occupied lattice cells match the instruction mask after local cross extension and cross-erasure, then recolor them.

    Input structure:
    - A black background is split by yellow separator lines into a 3x3 lattice of 5x5 cells.
    - Some non-instruction cells contain exactly one blue 2x2 solid block at one of several visible relative placements.
    - The bottom-right cell is an instruction region whose magenta pixels encode a 5x5 binary mask.

    Rule:
    - In every occupied non-instruction cell, keep the original 2x2 block and extend a blue full-width row and full-height column through that block's local anchor.
    - The same occupied cell also receives horizontal contributions from other occupied cells in its lattice row and vertical contributions from other occupied cells in its lattice column; any pixel where a horizontal source and a vertical source from different blocks cross is reset to black.
    - Compare each occupied cell's resulting blue-pixel set with the instruction mask exactly; if it matches, recolor those blue pixels to magenta, otherwise leave them blue. Grid lines, empty cells, and the instruction cell stay unchanged.

    Key insight:
    - Matching depends on the post-erasure residue of each occupied cell, not on the original 2x2 block alone, so shared lattice rows and columns indirectly control which cells satisfy the mask.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    row_starts, col_starts, blocks, instruction = _parse_blocks(g)
    mask = instruction == MAGENTA
    patterns = _processed_cells(blocks)
    return _render_output_from_patterns(g, row_starts, col_starts, patterns, mask)


def _base_grid() -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for sep in range(CELL_SIZE, GRID_SIZE, STRIDE):
        grid[sep, :] = YELLOW
        grid[:, sep] = YELLOW
    return grid


def _paint_block(cell: np.ndarray, top: int, left: int) -> None:
    cell[top : top + 2, left : left + 2] = BLUE


def _place_blocks(
    placements: Dict[Tuple[int, int], Tuple[int, int]],
) -> Tuple[np.ndarray, Dict[Tuple[int, int], Dict[str, int]], np.ndarray, Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], np.ndarray], List[int], List[int]]:
    grid = _base_grid()
    row_starts, col_starts = _parse_layout(grid)
    blocks: Dict[Tuple[int, int], Dict[str, int]] = {}
    for key, (top, left) in placements.items():
        cr, cc = key
        cell = _cell_view(grid, row_starts, col_starts, cr, cc)
        _paint_block(cell, top, left)
        blocks[key] = {
            "top": top,
            "left": left,
            "anchor_r": top + 1,
            "anchor_c": left + 1,
        }
    patterns = _processed_cells(blocks)
    own_only = _own_only_cells(blocks)
    return grid, blocks, np.zeros((CELL_SIZE, CELL_SIZE), dtype=bool), patterns, own_only, row_starts, col_starts


def _same_row_or_col(keys: Iterable[Tuple[int, int]]) -> bool:
    keys = list(keys)
    for i, (r1, c1) in enumerate(keys):
        for r2, c2 in keys[i + 1 :]:
            if r1 == r2 or c1 == c2:
                return True
    return False


def _candidate_cells() -> List[Tuple[int, int]]:
    cells = [(r, c) for r in range(GRID_CELLS) for c in range(GRID_CELLS)]
    cells.remove(INSTRUCTION_CELL)
    return cells


def _placement_features(blocks: Dict[Tuple[int, int], Dict[str, int]]) -> Dict[Tuple[int, int], Tuple[bool, bool]]:
    features: Dict[Tuple[int, int], Tuple[bool, bool]] = {}
    for key in blocks:
        cr, cc = key
        row_external = any(other != key and other[0] == cr for other in blocks)
        col_external = any(other != key and other[1] == cc for other in blocks)
        features[key] = (row_external, col_external)
    return features


def _semantic_checks(
    blocks: Dict[Tuple[int, int], Dict[str, int]],
    patterns: Dict[Tuple[int, int], np.ndarray],
    own_only: Dict[Tuple[int, int], np.ndarray],
    mask: np.ndarray,
) -> bool:
    if not (3 <= len(blocks) <= 5):
        return False
    if not _same_row_or_col(blocks):
        return False

    matching = [key for key, pattern in patterns.items() if np.array_equal(pattern == BLUE, mask)]
    if not matching or len(matching) == len(blocks):
        return False

    features = _placement_features(blocks)
    if not any(features[key][0] or features[key][1] for key in blocks):
        return False

    # At least one matching cell must rely on the interference, not only on its own block.
    if not any(
        np.array_equal(patterns[key] == BLUE, mask) and not np.array_equal(patterns[key], own_only[key])
        for key in blocks
    ):
        return False

    # At least one matching cell shares its original 2x2 placement with a non-matching cell.
    match_positions = {
        (blocks[key]["top"], blocks[key]["left"]): key
        for key in matching
    }
    if not any(
        key not in matching and (blocks[key]["top"], blocks[key]["left"]) in match_positions
        for key in blocks
    ):
        return False

    return True


def _choose_mask(
    blocks: Dict[Tuple[int, int], Dict[str, int]],
    patterns: Dict[Tuple[int, int], np.ndarray],
    own_only: Dict[Tuple[int, int], np.ndarray],
) -> np.ndarray | None:
    keys = list(blocks.keys())
    random.shuffle(keys)
    for key in keys:
        candidate = patterns[key] == BLUE
        if _semantic_checks(blocks, patterns, own_only, candidate):
            return candidate.copy()
    return None


def _render_input(
    placements: Dict[Tuple[int, int], Tuple[int, int]],
    mask: np.ndarray,
) -> np.ndarray:
    grid = _base_grid()
    row_starts, col_starts = _parse_layout(grid)
    for key, (top, left) in placements.items():
        cell = _cell_view(grid, row_starts, col_starts, *key)
        _paint_block(cell, top, left)

    instruction = _cell_view(grid, row_starts, col_starts, *INSTRUCTION_CELL)
    instruction[mask] = MAGENTA
    return grid


def _attempt() -> np.ndarray | None:
    block_count = random.randint(3, 5)
    chosen_cells = random.sample(_candidate_cells(), block_count)
    if not _same_row_or_col(chosen_cells):
        return None

    placements = {cell: random.choice(BLOCK_TOP_LEFTS) for cell in chosen_cells}
    grid, blocks, _, patterns, own_only, row_starts, col_starts = _place_blocks(placements)
    mask = _choose_mask(blocks, patterns, own_only)
    if mask is None:
        return None

    inp = _render_input(placements, mask)
    out = _render_output_from_patterns(inp, row_starts, col_starts, patterns, mask)

    # Validate using an independent parse from the actual rendered input.
    parsed_row_starts, parsed_col_starts, parsed_blocks, parsed_instruction = _parse_blocks(inp)
    parsed_patterns = _processed_cells(parsed_blocks)
    parsed_own_only = _own_only_cells(parsed_blocks)
    parsed_mask = parsed_instruction == MAGENTA
    if not _semantic_checks(parsed_blocks, parsed_patterns, parsed_own_only, parsed_mask):
        return None
    if not np.array_equal(parsed_row_starts, row_starts) or not np.array_equal(parsed_col_starts, col_starts):
        return None
    if not np.array_equal(out, _render_output_from_patterns(inp, parsed_row_starts, parsed_col_starts, parsed_patterns, parsed_mask)):
        return None
    if not np.array_equal(main(inp), out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
