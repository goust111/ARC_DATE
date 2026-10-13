from __future__ import annotations

from collections import Counter

import numpy as np


BLOCK_SIZE = 5
BLOCK_ROWS = 3
BLOCK_COLS = 3
GRID_SHAPE = (BLOCK_ROWS * BLOCK_SIZE, BLOCK_COLS * BLOCK_SIZE)

BLACK = 0
YELLOW = 4
ARM_PALETTE = np.array([1, 2, 3, 5, 6, 7, 8, 9], dtype=int)

COUNT_PATTERNS = {
    3: ((4, 2, 2), (4, 3, 1), (5, 2, 1)),
    4: ((3, 2, 2, 1), (4, 2, 1, 1)),
}
MAX_GENERATION_ATTEMPTS = 400


def _block_slice(block_r: int, block_c: int) -> tuple[slice, slice]:
    r0 = int(block_r) * BLOCK_SIZE
    c0 = int(block_c) * BLOCK_SIZE
    return slice(r0, r0 + BLOCK_SIZE), slice(c0, c0 + BLOCK_SIZE)


def _find_marker(block: np.ndarray) -> tuple[int, tuple[int, int]] | None:
    hits: list[tuple[int, tuple[int, int]]] = []
    for r in range(1, BLOCK_SIZE - 1):
        for c in range(1, BLOCK_SIZE - 1):
            if int(block[r, c]) != YELLOW:
                continue
            arms = [
                int(block[r - 1, c]),
                int(block[r + 1, c]),
                int(block[r, c - 1]),
                int(block[r, c + 1]),
            ]
            if len(set(arms)) != 1:
                continue
            arm_color = arms[0]
            if arm_color in (BLACK, YELLOW):
                continue
            hits.append((arm_color, (r, c)))
    assert len(hits) <= 1, "each 5x5 block must contain at most one valid marker"
    return hits[0] if hits else None


def _detect_blocks(grid: np.ndarray) -> list[dict[str, object]]:
    detections: list[dict[str, object]] = []
    for block_r in range(BLOCK_ROWS):
        for block_c in range(BLOCK_COLS):
            rs, cs = _block_slice(block_r, block_c)
            block = grid[rs, cs]
            marker = _find_marker(block)
            detections.append(
                {
                    "block_r": block_r,
                    "block_c": block_c,
                    "index": block_r * BLOCK_COLS + block_c,
                    "block": block.copy(),
                    "marker": marker,
                }
            )
    return detections


def _winner_color(detections: list[dict[str, object]]) -> int:
    counts = Counter(
        int(entry["marker"][0])
        for entry in detections
        if entry["marker"] is not None
    )
    ranked = counts.most_common()
    assert ranked, "expected at least one valid marker"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "winner arm color must be unique"
    return int(ranked[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the earliest block whose yellow-centered cross uses the globally winning arm color.

    Input structure:
    - The input is a 15x15 black grid partitioned into a 3x3 lattice of 5x5 blocks with no separator lines.
    - Exactly eight blocks contain one local marker: a yellow center pixel with four orthogonally adjacent arms
      of one shared non-black, non-yellow color; the remaining block has no valid marker and may contain only noise.
    - Blocks may also contain a few isolated noise pixels that do not form a valid marker.

    Rule:
    - Inspect each 5x5 block independently and detect whether it contains a valid yellow-centered cross marker.
    - Count the arm colors of all valid markers across the whole 3x3 block lattice and find the unique strictly
      most frequent arm color.
    - Among the blocks whose marker uses that winning arm color, return the complete 5x5 content of the earliest
      block in row-major order.

    Key insight:
    - The decisive information is split across two scales: marker recognition is local to each 5x5 block, but the
      winning color is determined only after aggregating those local readings globally.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == GRID_SHAPE, f"expected a {GRID_SHAPE[0]}x{GRID_SHAPE[1]} grid"

    detections = _detect_blocks(g)
    winner = _winner_color(detections)
    winner_blocks = [entry for entry in detections if entry["marker"] is not None and int(entry["marker"][0]) == winner]
    assert winner_blocks, "winner color must appear in at least one block"
    chosen = min(winner_blocks, key=lambda entry: (int(entry["block_r"]), int(entry["block_c"])))
    return np.array(chosen["block"], dtype=int, copy=True)


def _sample_counts(num_colors: int) -> list[int]:
    pattern = list(COUNT_PATTERNS[num_colors][np.random.randint(len(COUNT_PATTERNS[num_colors]))])
    counts = [int(pattern[0])]
    tail = [int(x) for x in pattern[1:] if int(x) > 0]
    np.random.shuffle(tail)
    counts.extend(tail)
    return counts


def _sample_color_assignment() -> tuple[list[int], list[int]]:
    num_colors = int(np.random.randint(3, 5))
    chosen = np.random.choice(ARM_PALETTE, size=num_colors, replace=False)
    winner_slot = int(np.random.randint(num_colors))
    colors = [int(chosen[winner_slot])]
    colors.extend(int(chosen[i]) for i in range(num_colors) if int(i) != winner_slot)
    return colors, _sample_counts(num_colors)


def _assign_marker_colors() -> tuple[list[int | None], int, int]:
    colors, counts = _sample_color_assignment()
    while True:
        invalid_index = int(np.random.randint(BLOCK_ROWS * BLOCK_COLS))
        valid_indices = [idx for idx in range(BLOCK_ROWS * BLOCK_COLS) if idx != invalid_index]
        winner_count = int(counts[0])
        winner_positions = set(int(x) for x in np.random.choice(valid_indices, size=winner_count, replace=False))
        earliest_winner = min(winner_positions)
        if any(idx < earliest_winner for idx in valid_indices if idx not in winner_positions):
            break

    assignments: list[int | None] = [None] * (BLOCK_ROWS * BLOCK_COLS)
    remaining = [idx for idx in valid_indices if idx not in winner_positions]
    np.random.shuffle(remaining)
    for idx in winner_positions:
        assignments[idx] = int(colors[0])

    cursor = 0
    for color, count in zip(colors[1:], counts[1:]):
        for idx in remaining[cursor : cursor + count]:
            assignments[idx] = int(color)
        cursor += count

    assert cursor == len(remaining)
    return assignments, invalid_index, int(colors[0])


def _marker_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {
        (r, c),
        (r - 1, c),
        (r + 1, c),
        (r, c - 1),
        (r, c + 1),
    }


def _available_noise_cells(block: np.ndarray, forbidden: set[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = [
        (r, c)
        for r in range(BLOCK_SIZE)
        for c in range(BLOCK_SIZE)
        if (r, c) not in forbidden and int(block[r, c]) == BLACK
    ]
    np.random.shuffle(cells)
    return cells


def _noise_budget(decoy_need: int) -> np.ndarray:
    budget = np.random.randint(0, 3, size=BLOCK_ROWS * BLOCK_COLS)
    while int(budget.sum()) < decoy_need + 2:
        growable = np.flatnonzero(budget < 2)
        if growable.size == 0:
            break
        budget[int(growable[np.random.randint(growable.size)])] += 1
    return budget


def _choose_decoy_color(marker_colors: list[int | None], winner_color: int) -> tuple[int, int]:
    cross_counts = Counter(int(color) for color in marker_colors if color is not None)
    losers = [(count, color) for color, count in cross_counts.items() if int(color) != winner_color]
    losers.sort(reverse=True)
    decoy_count, decoy_color = losers[0]
    decoy_need = 4 * (cross_counts[winner_color] - int(decoy_count))
    return int(decoy_color), int(decoy_need)


def _render_grid() -> np.ndarray:
    marker_colors, invalid_index, winner_color = _assign_marker_colors()
    decoy_color, decoy_need = _choose_decoy_color(marker_colors, winner_color)
    noise_budget = _noise_budget(decoy_need)
    if int(noise_budget.sum()) < decoy_need:
        raise AssertionError("insufficient noise slots")

    grid = np.zeros(GRID_SHAPE, dtype=int)
    forbidden_cells: list[set[tuple[int, int]]] = []

    for idx, color in enumerate(marker_colors):
        block = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=int)
        if color is None:
            forbidden_cells.append(set())
        else:
            center = tuple(int(x) for x in np.random.randint(1, BLOCK_SIZE - 1, size=2))
            cells = _marker_cells(center)
            block[center[0], center[1]] = YELLOW
            for r, c in cells - {center}:
                block[r, c] = int(color)
            forbidden_cells.append(cells)
        rs, cs = _block_slice(idx // BLOCK_COLS, idx % BLOCK_COLS)
        grid[rs, cs] = block

    noise_plan: list[list[int]] = [[] for _ in range(BLOCK_ROWS * BLOCK_COLS)]
    remaining = decoy_need
    block_order = list(np.random.permutation(BLOCK_ROWS * BLOCK_COLS))
    while remaining > 0:
        progressed = False
        for idx in block_order:
            if remaining == 0:
                break
            if len(noise_plan[idx]) >= int(noise_budget[idx]):
                continue
            noise_plan[idx].append(decoy_color)
            remaining -= 1
            progressed = True
        if not progressed:
            raise AssertionError("failed to allocate decoy noise")

    extra_colors = [int(c) for c in ARM_PALETTE if int(c) != decoy_color]
    for idx in range(BLOCK_ROWS * BLOCK_COLS):
        target = int(noise_budget[idx])
        while len(noise_plan[idx]) < target:
            marker_color = marker_colors[idx]
            if np.random.rand() < 0.7:
                noise_color = extra_colors[int(np.random.randint(len(extra_colors)))]
            elif marker_color is not None and int(marker_color) not in (winner_color, decoy_color):
                noise_color = int(marker_color)
            else:
                noise_color = extra_colors[int(np.random.randint(len(extra_colors)))]
            noise_plan[idx].append(noise_color)

    for idx, colors in enumerate(noise_plan):
        rs, cs = _block_slice(idx // BLOCK_COLS, idx % BLOCK_COLS)
        block = grid[rs, cs].copy()
        cells = _available_noise_cells(block, forbidden_cells[idx])
        if len(cells) < len(colors):
            raise AssertionError("not enough room for block noise")
        for (r, c), color in zip(cells, colors):
            block[r, c] = int(color)
        grid[rs, cs] = block

    assert invalid_index in range(BLOCK_ROWS * BLOCK_COLS)
    return grid


def _semantic_checks(grid: np.ndarray) -> None:
    detections = _detect_blocks(grid)
    valid = [entry for entry in detections if entry["marker"] is not None]
    invalid = [entry for entry in detections if entry["marker"] is None]
    assert len(valid) == 8, "exactly eight blocks must contain markers"
    assert len(invalid) == 1, "exactly one block must be marker-free"

    winner = _winner_color(detections)
    marker_counts = Counter(int(entry["marker"][0]) for entry in valid)
    assert marker_counts[winner] >= 2, "winner must appear multiple times"
    ranked = marker_counts.most_common()
    assert len(ranked) >= 2 and ranked[1][1] >= 2, "need a repeated non-winner color to rule out simpler shortcuts"

    winner_entries = [entry for entry in valid if int(entry["marker"][0]) == winner]
    winner_indices = [int(entry["index"]) for entry in winner_entries]
    earliest_winner = min(winner_indices)
    assert any(int(entry["index"]) < earliest_winner for entry in valid if int(entry["marker"][0]) != winner), "earliest valid block must not already be a winner"

    winner_blocks = {np.asarray(entry["block"], dtype=int).tobytes() for entry in winner_entries}
    assert len(winner_blocks) >= 2, "winner tie-break must matter on full block contents"

    colored_counts = Counter(int(v) for v in grid.ravel() if int(v) not in (BLACK, YELLOW))
    best_visible = max(colored_counts.values())
    assert colored_counts[winner] < best_visible or sum(1 for v in colored_counts.values() if v == best_visible) > 1, "winner should not be recoverable from raw pixel frequency alone"

    expected = main(grid)
    chosen = min(winner_entries, key=lambda entry: int(entry["index"]))
    assert np.array_equal(expected, np.asarray(chosen["block"], dtype=int))


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _render_grid()
        try:
            _semantic_checks(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
