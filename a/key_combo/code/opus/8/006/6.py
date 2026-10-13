from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4
TEAL = 8

PATTERN_SHAPE = (3, 3)
RETRY_LIMIT = 400


def _block_index(center_r: int, center_c: int, k: int) -> tuple[int, int]:
    return int(center_r // k), int(center_c // k)


def _matches_window(
    window: np.ndarray, skeleton_mask: np.ndarray, missing_mask: np.ndarray, fill_missing: bool
) -> bool:
    other_mask = ~(skeleton_mask | missing_mask)
    if not np.all(window[skeleton_mask] == BLUE):
        return False
    missing_color = YELLOW if fill_missing else BG
    if not np.all(window[missing_mask] == missing_color):
        return False
    return np.all(window[other_mask] == BG)


def _scan_matches(
    grid: np.ndarray, skeleton_mask: np.ndarray, missing_mask: np.ndarray
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    complete: list[tuple[int, int]] = []
    incomplete: list[tuple[int, int]] = []
    for r in range(h - 2):
        for c in range(w - 2):
            window = grid[r : r + 3, c : c + 3]
            if _matches_window(window, skeleton_mask, missing_mask, fill_missing=True):
                complete.append((r, c))
            elif _matches_window(window, skeleton_mask, missing_mask, fill_missing=False):
                incomplete.append((r, c))
    return complete, incomplete


def _find_template(
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    h, w = grid.shape
    assert h % 3 == 0 and w % 3 == 0 and h == w
    k = h // 3
    corners = ((0, 0), (0, 2), (2, 0), (2, 2))
    candidates: list[tuple[tuple[int, int, int, int], np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]] = []

    for br, bc in corners:
        r_start = br * k
        c_start = bc * k
        for r in range(r_start, r_start + k - 2):
            for c in range(c_start, c_start + k - 2):
                window = grid[r : r + 3, c : c + 3]
                colors = set(map(int, np.unique(window)))
                if not colors.issubset({BG, BLUE, YELLOW}):
                    continue
                skeleton_mask = window == BLUE
                missing_mask = window == YELLOW
                if int(np.count_nonzero(skeleton_mask)) < 2:
                    continue
                if int(np.count_nonzero(missing_mask)) < 2:
                    continue
                complete, incomplete = _scan_matches(grid, skeleton_mask, missing_mask)
                if (r, c) not in complete:
                    continue
                score = (
                    1 if len(complete) == 1 else 0,
                    len(incomplete),
                    len(complete) + len(incomplete),
                    -r,
                    -c,
                )
                candidates.append((score, skeleton_mask, missing_mask, (r, c), incomplete))

    assert candidates, "expected a corner template with blue skeleton and yellow mask"
    candidates.sort(reverse=True, key=lambda item: item[0])
    _score, skeleton_mask, missing_mask, template_pos, incomplete = candidates[0]
    return skeleton_mask, missing_mask, template_pos, incomplete


def _analyze(
    grid: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2 and h == w and h % 3 == 0
    k = h // 3

    gate = np.zeros((3, 3), dtype=bool)
    for br in range(3):
        for bc in range(3):
            block = g[br * k : (br + 1) * k, bc * k : (bc + 1) * k]
            gate[br, bc] = bool(np.any(block == RED))

    skeleton_mask, missing_mask, template_pos, incomplete = _find_template(g)
    return k, gate, skeleton_mask, missing_mask, template_pos, incomplete


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Compress red markers into a 3x3 gate that completes some template copies and erases others.

    Input structure:
    - The grid is a square 3Kx3K canvas with black background, partitioned into a 3x3 array of KxK coarse blocks.
    - Sparse red markers indicate which coarse blocks are active.
    - One corner coarse block contains a 3x3 template made of blue skeleton pixels and yellow missing pixels.
    - Other coarse blocks may contain incomplete 3x3 copies that keep the blue skeleton but replace the yellow cells by background.

    Rule:
    - Compress each coarse block to one gate bit: active iff the block contains any red marker.
    - Read the template's blue cells as the skeleton mask and its yellow cells as the missing mask.
    - Scan every 3x3 window; windows that match the skeleton exactly and leave all missing-mask cells black are incomplete copies.
    - In active coarse blocks, fill each incomplete copy's missing-mask cells with yellow.
    - In silent coarse blocks, erase every detected copy window; the complete template itself is treated the same way based on its block.

    Key insight:
    - The red-marker compression is not the output; it is a spatial gate that routes the same template match either to completion or to deletion.
    """
    g = np.asarray(input_grid, dtype=int)
    out = g.copy()
    k, gate, skeleton_mask, missing_mask, template_pos, incomplete = _analyze(g)

    def apply_window(top_left: tuple[int, int], is_complete: bool) -> None:
        r, c = top_left
        br, bc = _block_index(r + 1, c + 1, k)
        active = bool(gate[br, bc])
        if active:
            if not is_complete:
                out[r : r + 3, c : c + 3][missing_mask] = YELLOW
        else:
            erase_mask = skeleton_mask | missing_mask if is_complete else skeleton_mask
            out[r : r + 3, c : c + 3][erase_mask] = BG

    apply_window(template_pos, is_complete=True)
    for pos in incomplete:
        apply_window(pos, is_complete=False)

    return out


def _random_template() -> np.ndarray:
    while True:
        patch = np.zeros(PATTERN_SHAPE, dtype=int)
        cells = list(range(9))
        random.shuffle(cells)
        blue_count = random.randint(2, 4)
        yellow_count = random.randint(2, min(4, 9 - blue_count))
        blue_cells = cells[:blue_count]
        yellow_cells = cells[blue_count : blue_count + yellow_count]
        for idx in blue_cells:
            patch[idx // 3, idx % 3] = BLUE
        for idx in yellow_cells:
            patch[idx // 3, idx % 3] = YELLOW
        if np.count_nonzero(patch == BG) == 0:
            continue
        if random.random() < 0.5:
            patch = np.rot90(patch, random.randint(0, 3))
        if random.random() < 0.5:
            patch = np.fliplr(patch)
        if random.random() < 0.5:
            patch = np.flipud(patch)
        return patch


def _window_candidates(block_r: int, block_c: int, k: int) -> list[tuple[int, int]]:
    return [
        (block_r * k + dr, block_c * k + dc)
        for dr in range(k - 2)
        for dc in range(k - 2)
    ]


def _place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    top_left: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> bool:
    r, c = top_left
    cells = {(r + dr, c + dc) for dr in range(3) for dc in range(3)}
    if cells & occupied:
        return False
    if any(not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]) for rr, cc in cells):
        return False
    if any(grid[rr, cc] != BG for rr, cc in cells):
        return False
    grid[r : r + 3, c : c + 3] = patch
    occupied |= cells
    return True


def _free_cells_in_block(
    grid: np.ndarray, block_r: int, block_c: int, k: int, occupied: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(block_r * k, (block_r + 1) * k):
        for c in range(block_c * k, (block_c + 1) * k):
            if grid[r, c] == BG and (r, c) not in occupied:
                cells.append((r, c))
    return cells


def _build_input() -> np.ndarray | None:
    k = random.choice((4, 5))
    size = 3 * k
    grid = np.zeros((size, size), dtype=int)
    occupied: set[tuple[int, int]] = set()

    block_ids = [(br, bc) for br in range(3) for bc in range(3)]
    active_count = random.randint(4, 6)
    active_blocks = set(random.sample(block_ids, active_count))

    corner_blocks = [(0, 0), (0, 2), (2, 0), (2, 2)]
    template_block = random.choice(corner_blocks)

    incomplete_count = random.randint(4, 6)
    candidate_copy_blocks = [bid for bid in block_ids if bid != template_block]
    copy_blocks = random.sample(candidate_copy_blocks, incomplete_count)
    active_copy_blocks = [bid for bid in copy_blocks if bid in active_blocks]
    silent_copy_blocks = [bid for bid in copy_blocks if bid not in active_blocks]
    if not active_copy_blocks or not silent_copy_blocks:
        return None

    template = _random_template()
    incomplete_patch = np.where(template == YELLOW, BG, template)

    template_positions = _window_candidates(template_block[0], template_block[1], k)
    random.shuffle(template_positions)
    placed_template = False
    template_pos = (-1, -1)
    for pos in template_positions:
        if _place_patch(grid, template, pos, occupied):
            placed_template = True
            template_pos = pos
            break
    if not placed_template:
        return None

    placed_copy_blocks: list[tuple[int, int]] = []
    for block_r, block_c in copy_blocks:
        positions = _window_candidates(block_r, block_c, k)
        random.shuffle(positions)
        for pos in positions:
            if _place_patch(grid, incomplete_patch, pos, occupied):
                placed_copy_blocks.append((block_r, block_c))
                break
        else:
            return None

    if len(placed_copy_blocks) != incomplete_count:
        return None

    for block_r, block_c in active_blocks:
        free_cells = _free_cells_in_block(grid, block_r, block_c, k, occupied)
        if len(free_cells) == 0:
            return None
        marker_count = random.randint(1, min(2, len(free_cells)))
        for rr, cc in random.sample(free_cells, marker_count):
            grid[rr, cc] = RED
            occupied.add((rr, cc))

    empty_blocks = [bid for bid in block_ids if bid != template_block and bid not in copy_blocks]
    random.shuffle(empty_blocks)
    distractor_total = random.randint(1, min(3, len(empty_blocks))) if empty_blocks else 0
    for block_r, block_c in empty_blocks[:distractor_total]:
        free_cells = _free_cells_in_block(grid, block_r, block_c, k, occupied)
        if not free_cells:
            continue
        rr, cc = random.choice(free_cells)
        grid[rr, cc] = random.choice((BLUE, TEAL))
        occupied.add((rr, cc))

    try:
        analyzed = _analyze(grid)
    except Exception:
        return None

    analyzed_k, gate, skeleton_mask, missing_mask, found_template_pos, incomplete = analyzed
    if analyzed_k != k:
        return None
    if found_template_pos != template_pos:
        return None
    if len(incomplete) != incomplete_count:
        return None

    incomplete_blocks_found = {
        _block_index(r + 1, c + 1, k)
        for r, c in incomplete
    }
    if len(incomplete_blocks_found) != incomplete_count:
        return None
    if {_block_index(template_pos[0] + 1, template_pos[1] + 1, k)} != {template_block}:
        return None
    if not any(gate[br, bc] for br, bc in incomplete_blocks_found):
        return None
    if not any(not gate[br, bc] for br, bc in incomplete_blocks_found):
        return None

    output = main(grid)
    template_window_out = output[template_pos[0] : template_pos[0] + 3, template_pos[1] : template_pos[1] + 3]
    if gate[template_block]:
        if not np.array_equal(template_window_out, template):
            return None
    else:
        if np.any(template_window_out[skeleton_mask | missing_mask] != BG):
            return None

    completed = 0
    erased = 0
    for r, c in incomplete:
        block_r, block_c = _block_index(r + 1, c + 1, k)
        window_out = output[r : r + 3, c : c + 3]
        if gate[block_r, block_c]:
            if not _matches_window(window_out, skeleton_mask, missing_mask, fill_missing=True):
                return None
            completed += 1
        else:
            if np.any(window_out[skeleton_mask] != BG):
                return None
            erased += 1
    if completed == 0 or erased == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated retries")
