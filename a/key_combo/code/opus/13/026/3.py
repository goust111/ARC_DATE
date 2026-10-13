from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
LABEL_COLORS = (1, 2, 3, 4)
GRID_SIZE_RANGE = (10, 13)
GENERATION_RETRIES = 500
BLOCK_DIMS = (
    (1, 1),
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 2),
    (1, 4),
    (4, 1),
    (2, 3),
    (3, 2),
)


def _expand_runs(runs: list[tuple[int, int]]) -> np.ndarray:
    seq: list[int] = []
    for color, length in runs:
        seq.extend([int(color)] * int(length))
    return np.array(seq, dtype=int)


def _ordered_runs(
    runs: list[tuple[int, int]],
    edge_color: int | None = None,
) -> list[tuple[int, int]]:
    ordered = runs[:]
    random.shuffle(ordered)
    if edge_color is None:
        return ordered

    edge_run = next(run for run in ordered if int(run[0]) == int(edge_color))
    others = [run for run in ordered if int(run[0]) != int(edge_color)]
    if random.random() < 0.5:
        return [edge_run, *others]
    return [*others, edge_run]


def _match_mask(row_labels: np.ndarray, col_labels: np.ndarray) -> np.ndarray:
    return row_labels[:, None] == col_labels[None, :]


def _has_adjacent_match(mask: np.ndarray) -> bool:
    coords = {tuple(map(int, cell)) for cell in np.argwhere(mask)}
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if (r + dr, c + dc) in coords:
                    return True
    return False


def _has_fused_halo(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            nearby = 0
            for rr in range(max(0, r - 1), min(h, r + 2)):
                for cc in range(max(0, c - 1), min(w, c + 2)):
                    if mask[rr, cc]:
                        nearby += 1
                        if nearby >= 2:
                            return True
    return False


def _touches_interior_edge(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :].any()
        or mask[-1, :].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
    )


def main(grid: np.ndarray) -> np.ndarray:
    """Fill same-label intersections, then add a gray one-step halo around each filled interior cell.

    Input structure
    - The interior rectangle is black, with a right-edge row-label column and a bottom-edge column-label row.
    - Row labels and column labels use non-black colors from the same visible palette, and the bottom-right corner stays black.

    Rule
    - For each interior cell, compare its row label and column label; if they match, fill that cell with the shared label color.
    - For every filled match cell, recolor each still-black interior cell in its Chebyshev-distance-1 neighborhood to gray.
    - Do not overwrite colored match cells, the label row, the label column, or the bottom-right corner.

    Key insight
    - The colored cells are the equality table of the border labels, and those dynamically discovered match positions become the centers of the gray halo stamps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 2 and w >= 2, "expected a nonempty interior plus border labels"

    row_labels = g[:-1, -1]
    col_labels = g[-1, :-1]

    assert np.all(g[:-1, :-1] == BG), "expected a black interior"
    assert np.all(np.isin(row_labels, LABEL_COLORS)), "expected non-black row labels"
    assert np.all(np.isin(col_labels, LABEL_COLORS)), "expected non-black column labels"
    assert int(g[-1, -1]) == BG, "expected a black bottom-right corner"

    out = g.copy()
    matches = _match_mask(row_labels, col_labels)
    out[:-1, :-1] = np.where(matches, row_labels[:, None], BG)

    centers = np.argwhere(matches)
    for r, c in centers:
        r = int(r)
        c = int(c)
        for rr in range(max(0, r - 1), min(h - 1, r + 2)):
            for cc in range(max(0, c - 1), min(w - 1, c + 2)):
                if out[rr, cc] == BG:
                    out[rr, cc] = GREY

    return out


def _sample_blocks(interior_h: int, interior_w: int) -> list[tuple[int, int]] | None:
    shared_count = 1 if random.random() < 0.55 else 2
    dims = [
        (r, c)
        for r, c in BLOCK_DIMS
        if r <= interior_h - 1 and c <= interior_w - 1
    ]
    if not dims:
        return None

    for _ in range(100):
        blocks = [random.choice(dims) for _ in range(shared_count)]
        areas = [r * c for r, c in blocks]
        total = sum(areas)
        if not (3 <= total <= 8):
            continue
        if not any(area >= 2 for area in areas):
            continue
        if sum(r for r, _ in blocks) >= interior_h:
            continue
        if sum(c for _, c in blocks) >= interior_w:
            continue
        return blocks
    return None


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    interior_h = height - 1
    interior_w = width - 1

    blocks = _sample_blocks(interior_h, interior_w)
    if blocks is None:
        return None

    shared_count = len(blocks)
    colors = random.sample(LABEL_COLORS, shared_count + 2)
    shared_colors = colors[:shared_count]
    row_only_color = colors[shared_count]
    col_only_color = colors[shared_count + 1]

    row_runs = [(shared_colors[i], blocks[i][0]) for i in range(shared_count)]
    row_runs.append((row_only_color, interior_h - sum(length for _color, length in row_runs)))
    col_runs = [(shared_colors[i], blocks[i][1]) for i in range(shared_count)]
    col_runs.append((col_only_color, interior_w - sum(length for _color, length in col_runs)))

    edge_color = random.choice(shared_colors)
    force_row_edge = random.random() < 0.6
    force_col_edge = random.random() < 0.6

    row_labels = _expand_runs(_ordered_runs(row_runs, edge_color if force_row_edge else None))
    col_labels = _expand_runs(_ordered_runs(col_runs, edge_color if force_col_edge else None))

    grid = np.full((height, width), BG, dtype=int)
    grid[:-1, -1] = row_labels
    grid[-1, :-1] = col_labels

    matches = _match_mask(row_labels, col_labels)
    match_count = int(matches.sum())
    if not (3 <= match_count <= 8):
        return None
    if not _has_adjacent_match(matches):
        return None
    if not _has_fused_halo(matches):
        return None

    out = main(grid)
    halo_cells = int(np.count_nonzero(out[:-1, :-1] == GREY))
    if halo_cells == 0:
        return None
    if halo_cells > (interior_h * interior_w) // 2:
        return None
    if (force_row_edge or force_col_edge) and not _touches_interior_edge(matches):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
