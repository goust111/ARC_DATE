import math
import random

import numpy as np


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _split_blocks(mask: np.ndarray, n: int) -> np.ndarray:
    return np.array(
        [[mask[br * n : (br + 1) * n, bc * n : (bc + 1) * n] for bc in range(n)] for br in range(n)],
        dtype=np.uint8,
    )


def _expand_binary(mother: np.ndarray) -> np.ndarray:
    n = int(mother.shape[0])
    out = np.zeros((n * n, n * n), dtype=np.uint8)
    for br in range(n):
        for bc in range(n):
            if int(mother[br, bc]) == 1:
                out[br * n : (br + 1) * n, bc * n : (bc + 1) * n] = mother
    return out


def _raw_content_majority(blocks: np.ndarray, observed_layout: np.ndarray, tie_fill: int | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    filled_coords = np.argwhere(observed_layout == 1)
    n = int(observed_layout.shape[0])
    if len(filled_coords) == 0:
        return np.zeros((n, n), dtype=np.uint8), np.zeros((n, n), dtype=bool)

    stack = np.array([blocks[int(br), int(bc)] for br, bc in filled_coords], dtype=np.uint8)
    ones = stack.sum(axis=0)
    zeros = len(filled_coords) - ones
    ties = ones == zeros
    majority = np.where(ones > zeros, 1, 0).astype(np.uint8)
    if np.isscalar(tie_fill):
        majority[ties] = int(tie_fill)
    else:
        majority[ties] = np.asarray(tie_fill, dtype=np.uint8)[ties]
    return majority, ties


def _classify_copy_blocks(blocks: np.ndarray, mother: np.ndarray, observed_layout: np.ndarray) -> np.ndarray:
    n = int(mother.shape[0])
    filled = np.zeros((n, n), dtype=np.uint8)
    for br in range(n):
        for bc in range(n):
            block = blocks[br, bc]
            dist_copy = int(np.count_nonzero(block != mother))
            dist_empty = int(block.sum())
            if dist_copy < dist_empty or (dist_copy == dist_empty and int(mother[br, bc]) == 1 and int(observed_layout[br, bc]) == 1):
                filled[br, bc] = 1
    if not filled.any():
        return observed_layout.copy()
    return filled


def _refine_mother(blocks: np.ndarray, observed_layout: np.ndarray, seed: np.ndarray) -> np.ndarray:
    mother = np.asarray(seed, dtype=np.uint8).copy()
    for _ in range(12):
        copy_layout = _classify_copy_blocks(blocks, mother, observed_layout)
        filled_coords = np.argwhere(copy_layout == 1)
        stack = np.array([blocks[int(br), int(bc)] for br, bc in filled_coords], dtype=np.uint8)
        content_ones = stack.sum(axis=0)
        ones = content_ones + copy_layout
        zeros = (len(filled_coords) - content_ones) + (1 - copy_layout)
        new_mother = np.where(ones > zeros, 1, np.where(ones < zeros, 0, observed_layout)).astype(np.uint8)
        if np.array_equal(new_mother, mother):
            break
        mother = new_mother
    return mother


def _candidate_score(blocks: np.ndarray, observed_layout: np.ndarray, mother: np.ndarray) -> int:
    copy_layout = _classify_copy_blocks(blocks, mother, observed_layout)
    layout_weight = max(2, int(observed_layout.sum()))
    penalty = layout_weight * int(np.count_nonzero(copy_layout != mother))
    for br in range(mother.shape[0]):
        for bc in range(mother.shape[1]):
            block = blocks[br, bc]
            if int(copy_layout[br, bc]) == 1:
                penalty += int(np.count_nonzero(block != mother))
            else:
                penalty += int(block.sum())
    return penalty


def _recover_mother_binary(binary_grid: np.ndarray) -> np.ndarray:
    h, w = binary_grid.shape
    assert h == w, "expected a square grid"
    n = math.isqrt(h)
    assert n * n == h, "expected side length to be a perfect square"

    blocks = _split_blocks(binary_grid, n)
    observed_layout = (blocks.sum(axis=(2, 3)) > 0).astype(np.uint8)
    assert observed_layout.any(), "expected at least one non-empty coarse block"

    content_occ_tiebreak, _ = _raw_content_majority(blocks, observed_layout, observed_layout)
    content_zero_tiebreak, _ = _raw_content_majority(blocks, observed_layout, 0)
    content_one_tiebreak, _ = _raw_content_majority(blocks, observed_layout, 1)

    seeds = [
        observed_layout,
        content_occ_tiebreak,
        content_zero_tiebreak,
        content_one_tiebreak,
    ]
    refined = [_refine_mother(blocks, observed_layout, seed) for seed in seeds]
    best = min(refined, key=lambda mother: _candidate_score(blocks, observed_layout, mother))
    return best.astype(np.uint8)


def main(grid: np.ndarray) -> np.ndarray:
    """Restore the unique mother sprite of a noisy self-expansion, then re-expand it exactly.

    Input structure:
    - The grid is square, and its side length is a perfect square `n^2`, so it can be partitioned into an `n x n` array of `n x n` coarse blocks.
    - The background is dominant and there is only one foreground color.
    - In the clean pattern, each foreground mother cell marks a coarse block that contains a full copy of that same mother pattern, while all other coarse blocks are empty.

    Rule:
    - Interpret every non-empty coarse block as a noisy vote for the hidden `n x n` mother pattern, and interpret the coarse-block occupancy map as another noisy view of that same mother.
    - Repeatedly keep the blocks that are more copy-like than empty, take a cellwise majority over their contents, and cross-check that result against the recovered occupancy layout.
    - Once the unique self-consistent mother pattern is recovered, regenerate the exact self-expansion by stamping that mother into every coarse block whose mother cell is foreground.

    Key insight:
    - The template is never shown separately: the block contents and the block-placement mask are redundant encodings of the same sprite, so they can correct each other's noise.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _majority_color(g)
    fg_colors = [int(v) for v in np.unique(g) if int(v) != bg]
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    fg = fg_colors[0]

    mother = _recover_mother_binary((g == fg).astype(np.uint8))
    expanded = _expand_binary(mother)
    return np.where(expanded == 1, fg, bg).astype(int)


# --------------- generator ---------------

BACKGROUND_COLOR = 0
FOREGROUND_COLORS = tuple(range(1, 10))
SIDE_OPTIONS = (3, 4, 5)
MODE_WEIGHTS = (("erase", 2), ("mixed", 3))
MAX_GENERATION_ATTEMPTS = 500


def _weighted_mode_choice() -> str:
    pool = [mode for mode, weight in MODE_WEIGHTS for _ in range(weight)]
    return random.choice(pool)


def _sample_mother(n: int, fg_count: int) -> np.ndarray:
    cells = [(r, c) for r in range(n) for c in range(n)]
    for _ in range(300):
        chosen = set(random.sample(cells, fg_count))
        rows_used = {r for r, _c in chosen}
        cols_used = {c for _r, c in chosen}
        if len(rows_used) < min(2, n) or len(cols_used) < min(2, n):
            continue
        mother = np.zeros((n, n), dtype=np.uint8)
        for r, c in chosen:
            mother[r, c] = 1
        return mother
    raise RuntimeError("failed to sample a non-degenerate mother pattern")


def _apply_erase_noise(binary_grid: np.ndarray, mother: np.ndarray) -> tuple[np.ndarray, dict]:
    n = int(mother.shape[0])
    noisy = binary_grid.copy()
    filled_positions = [(r, c) for r, c in np.argwhere(mother == 1)]
    erase_r, erase_c = random.choice(filled_positions)
    block = noisy[erase_r * n : (erase_r + 1) * n, erase_c * n : (erase_c + 1) * n]
    block[mother == 1] = 0
    return noisy, {"erased_block": (erase_r, erase_c)}


def _apply_mixed_noise(binary_grid: np.ndarray, mother: np.ndarray) -> tuple[np.ndarray, dict] | tuple[None, None]:
    n = int(mother.shape[0])
    filled_positions = [(r, c) for r, c in np.argwhere(mother == 1)]
    empty_positions = [(r, c) for r, c in np.argwhere(mother == 0)]
    if len(empty_positions) < len(filled_positions):
        return None, None

    noisy, erase_meta = _apply_erase_noise(binary_grid, mother)
    q_r, q_c = random.choice(empty_positions)
    spurious_hosts = [pos for pos in empty_positions if pos != (q_r, q_c)]
    if len(spurious_hosts) < len(filled_positions) - 1:
        return None, None

    selected_hosts = random.sample(spurious_hosts, len(filled_positions) - 1)
    for host_r, host_c in selected_hosts:
        noisy[host_r * n + q_r, host_c * n + q_c] = 1

    meta = {
        **erase_meta,
        "noise_cell": (q_r, q_c),
        "spurious_hosts": selected_hosts,
    }
    return noisy, meta


def _shortcut_masks(binary_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = math.isqrt(binary_grid.shape[0])
    blocks = _split_blocks(binary_grid, n)
    observed_layout = (blocks.sum(axis=(2, 3)) > 0).astype(np.uint8)
    content_occ, _ = _raw_content_majority(blocks, observed_layout, observed_layout)
    content_zero, ties = _raw_content_majority(blocks, observed_layout, 0)
    content_one, _ = _raw_content_majority(blocks, observed_layout, 1)
    return observed_layout, content_occ, content_zero, content_one


def _validate_semantics(noisy_binary: np.ndarray, mother: np.ndarray, ideal_binary: np.ndarray, mode: str) -> bool:
    observed_layout, content_occ, content_zero, content_one = _shortcut_masks(noisy_binary)
    if np.array_equal(noisy_binary, ideal_binary):
        return False
    if mode == "erase":
        if np.array_equal(observed_layout, mother):
            return False
        if np.array_equal(_expand_binary(observed_layout), ideal_binary):
            return False
    elif mode == "mixed":
        if np.array_equal(observed_layout, mother):
            return False
        if np.array_equal(content_zero, mother):
            return False
        if np.array_equal(content_one, mother):
            return False
        if np.array_equal(_expand_binary(observed_layout), ideal_binary):
            return False
        if np.array_equal(_expand_binary(content_zero), ideal_binary):
            return False
        if np.array_equal(_expand_binary(content_one), ideal_binary):
            return False
        if np.array_equal(content_occ, mother):
            return False
    recovered = _recover_mother_binary(noisy_binary)
    return np.array_equal(recovered, mother)


def _gen_attempt() -> np.ndarray | None:
    mode = _weighted_mode_choice()
    n = random.choice(SIDE_OPTIONS)

    if mode == "erase":
        max_fg = min(n + 2, n * n - 3)
        fg_count = random.randint(3, max_fg)
    else:
        fg_count = random.choice((3, 4))
        if n * n - fg_count < fg_count:
            return None

    mother = _sample_mother(n, fg_count)
    ideal_binary = _expand_binary(mother)

    if mode == "erase":
        noisy_binary, _meta = _apply_erase_noise(ideal_binary, mother)
    else:
        noisy_binary, _meta = _apply_mixed_noise(ideal_binary, mother)
        if noisy_binary is None:
            return None

    if not _validate_semantics(noisy_binary, mother, ideal_binary, mode):
        return None

    fg = random.choice(FOREGROUND_COLORS)
    noisy_grid = np.where(noisy_binary == 1, fg, BACKGROUND_COLOR).astype(int)
    ideal_grid = np.where(ideal_binary == 1, fg, BACKGROUND_COLOR).astype(int)
    if not np.array_equal(main(noisy_grid), ideal_grid):
        return None
    return noisy_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid self-referential restoration input")
