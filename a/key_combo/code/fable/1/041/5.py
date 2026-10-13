from __future__ import annotations

from collections import Counter
import random

import numpy as np


PALETTE = tuple(range(10))
MAX_ATTEMPTS = 400


def _group_consecutive(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


def _full_color_lines(grid: np.ndarray, axis: int) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    if axis == 0:
        for r in range(grid.shape[0]):
            color = int(grid[r, 0])
            if np.all(grid[r] == color):
                out.setdefault(color, []).append(r)
    else:
        for c in range(grid.shape[1]):
            color = int(grid[0, c])
            if np.all(grid[:, c] == color):
                out.setdefault(color, []).append(c)
    return out


def _extract_cells(grid: np.ndarray) -> tuple[int, list[np.ndarray], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    row_lines = _full_color_lines(g, axis=0)
    col_lines = _full_color_lines(g, axis=1)

    candidates: list[tuple[int, list[np.ndarray], tuple[int, int]]] = []
    for color in sorted(set(row_lines) & set(col_lines)):
        row_runs = _group_consecutive(row_lines[color])
        col_runs = _group_consecutive(col_lines[color])
        if len(row_runs) < 3 or len(col_runs) < 3:
            continue
        row_thicknesses = {r1 - r0 + 1 for r0, r1 in row_runs}
        col_thicknesses = {c1 - c0 + 1 for c0, c1 in col_runs}
        if len(row_thicknesses) != 1 or len(col_thicknesses) != 1:
            continue
        if row_runs[0][0] != 0 or row_runs[-1][1] != H - 1:
            continue
        if col_runs[0][0] != 0 or col_runs[-1][1] != W - 1:
            continue

        row_gaps = [row_runs[i + 1][0] - row_runs[i][1] - 1 for i in range(len(row_runs) - 1)]
        col_gaps = [col_runs[i + 1][0] - col_runs[i][1] - 1 for i in range(len(col_runs) - 1)]
        if any(gap <= 0 for gap in row_gaps + col_gaps):
            continue
        if len(set(row_gaps)) != 1 or len(set(col_gaps)) != 1:
            continue

        cell_h = int(row_gaps[0])
        cell_w = int(col_gaps[0])
        cells: list[np.ndarray] = []
        ok = True
        for rr in range(len(row_runs) - 1):
            top = row_runs[rr][1] + 1
            bottom = row_runs[rr + 1][0]
            for cc in range(len(col_runs) - 1):
                left = col_runs[cc][1] + 1
                right = col_runs[cc + 1][0]
                patch = g[top:bottom, left:right]
                if patch.shape != (cell_h, cell_w):
                    ok = False
                    break
                cells.append(patch.copy())
            if not ok:
                break
        if ok:
            candidates.append((color, cells, (cell_h, cell_w)))

    assert len(candidates) == 1, "expected exactly one uniform lattice"
    return candidates[0]


def _majority_color(values: np.ndarray) -> int:
    counts = Counter(int(v) for v in values)
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    assert len(winners) == 1, "majority vote must be unambiguous"
    return winners[0]


def _majority_patch(cells: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(cells, axis=0)
    n, h, w = stack.shape
    out = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = _majority_color(stack[:, r, c])
    return out


def _mirror_scores(cells: list[np.ndarray], canonical: np.ndarray) -> list[int]:
    mirrored = np.fliplr(canonical)
    return [
        int(np.count_nonzero(cell != canonical) - np.count_nonzero(cell != mirrored))
        for cell in cells
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the clean sprite by majority vote and return the clean horizontal mirror hidden in one lattice cell.

    Input structure:
    - Full-row and full-column separator lines form a uniform lattice of same-sized cells.
    - Every cell contains a noisy copy of the same cell-sized sprite pattern.
    - Exactly one cell instead contains the noisy left-right mirror of that pattern, and the clean pattern is not mirror-symmetric.

    Rule:
    - Detect the lattice and extract every cell without the separator lines.
    - Take a per-pixel majority vote across all cells to reconstruct the clean canonical cell.
    - Compare each noisy cell against both the canonical cell and its left-right mirror; the impostor is the unique cell closer to the mirrored version.
    - Output the clean mirrored cell only, with the separator lines removed.

    Key insight:
    - Sparse recolor noise changes a few pixels, while the impostor differs in a coherent mirrored layout, so majority recovery must happen before mirror detection.
    """
    _line_color, cells, cell_shape = _extract_cells(np.asarray(grid, dtype=int))
    assert len(cells) >= 3

    canonical = _majority_patch(cells)
    mirrored = np.fliplr(canonical)
    assert canonical.shape == cell_shape
    assert np.any(canonical != mirrored), "clean sprite must be asymmetric"

    scores = _mirror_scores(cells, canonical)
    best = int(np.argmax(scores))
    sorted_scores = sorted(scores, reverse=True)
    assert sorted_scores[0] > 0, "impostor must be closer to the mirrored template"
    if len(sorted_scores) > 1:
        assert sorted_scores[0] > sorted_scores[1], "mirrored impostor must be unique"
    return mirrored


def _random_connected_mask(height: int, width: int) -> np.ndarray:
    box_h = random.randint(max(2, height - 3), height)
    box_w = random.randint(max(2, width - 3), width)
    top = random.randint(0, height - box_h)
    left = random.randint(0, width - box_w)

    box_area = box_h * box_w
    area_floor = min(box_area - 1, max(3, box_area // 4))
    area_ceiling = min(box_area - 1, max(area_floor, (box_area * 3) // 5))
    target_area = random.randint(area_floor, area_ceiling)

    cells = {(random.randrange(box_h), random.randrange(box_w))}
    while len(cells) < target_area:
        frontier = set()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < box_h and 0 <= cc < box_w and (rr, cc) not in cells:
                    frontier.add((rr, cc))
        cells.add(random.choice(tuple(frontier)))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[top + r, left + c] = True
    return mask


def _decorate_mask(mask: np.ndarray, bg: int, fg_colors: list[int]) -> np.ndarray:
    cell = np.full(mask.shape, bg, dtype=int)
    coords = [tuple(rc) for rc in np.argwhere(mask)]
    random.shuffle(coords)
    for r, c in coords:
        cell[r, c] = fg_colors[0]

    cursor = 0
    for color in fg_colors[1:]:
        accent_limit = max(1, len(coords) // 5)
        accent_count = random.randint(1, min(accent_limit, len(coords) - cursor))
        for r, c in coords[cursor : cursor + accent_count]:
            cell[r, c] = color
        cursor += accent_count
        if cursor >= len(coords):
            break
    return cell


def _make_canonical_cell(cell_h: int, cell_w: int, bg: int, fg_colors: list[int]) -> np.ndarray | None:
    for _ in range(80):
        mask = _random_connected_mask(cell_h, cell_w)
        cell = _decorate_mask(mask, bg, fg_colors)
        mirrored = np.fliplr(cell)
        asymmetry = int(np.count_nonzero(cell != mirrored))
        if asymmetry < max(6, cell_w):
            continue
        if np.any(cell[:, 0] != bg) and np.any(cell[:, -1] != bg):
            continue
        return cell
    return None


def _add_noise(base: np.ndarray, available_colors: list[int], min_flips: int, max_flips: int) -> np.ndarray:
    out = base.copy()
    flat_indices = random.sample(range(base.size), k=random.randint(min_flips, max_flips))
    width = base.shape[1]
    for idx in flat_indices:
        r, c = divmod(idx, width)
        choices = [color for color in available_colors if color != int(out[r, c])]
        out[r, c] = random.choice(choices)
    return out


def _build_grid(cells: list[np.ndarray], line_color: int, line_thickness: int) -> np.ndarray:
    rows = len(cells)
    cols = len(cells[0])
    cell_h, cell_w = cells[0][0].shape
    height = rows * cell_h + (rows + 1) * line_thickness
    width = cols * cell_w + (cols + 1) * line_thickness
    grid = np.full((height, width), line_color, dtype=int)

    for rr in range(rows):
        for cc in range(cols):
            top = line_thickness + rr * (cell_h + line_thickness)
            left = line_thickness + cc * (cell_w + line_thickness)
            grid[top : top + cell_h, left : left + cell_w] = cells[rr][cc]
    return grid


def _gen_attempt() -> np.ndarray | None:
    lattice_rows, lattice_cols = random.choice(((3, 3), (3, 5), (5, 3), (5, 5)))
    line_thickness = random.randint(1, 2)

    max_cell_h = (30 - (lattice_rows + 1) * line_thickness) // lattice_rows
    max_cell_w = (30 - (lattice_cols + 1) * line_thickness) // lattice_cols
    if max_cell_h < 4 or max_cell_w < 4:
        return None
    cell_h = random.randint(4, min(7, max_cell_h))
    cell_w = random.randint(4, min(7, max_cell_w))

    height = lattice_rows * cell_h + (lattice_rows + 1) * line_thickness
    width = lattice_cols * cell_w + (lattice_cols + 1) * line_thickness
    if height > 30 or width > 30:
        return None

    colors = random.sample(PALETTE, k=5)
    line_color = colors[0]
    cell_bg = colors[1]
    fg_count = random.randint(1, 3)
    fg_colors = colors[2 : 2 + fg_count]
    noise_palette = [color for color in PALETTE if color != line_color]

    canonical = _make_canonical_cell(cell_h, cell_w, cell_bg, fg_colors)
    if canonical is None:
        return None
    expected = np.fliplr(canonical)

    mirror_index = random.randrange(lattice_rows * lattice_cols)
    max_noise = min(3, max(2, (cell_h * cell_w) // 8))
    noisy_rows: list[list[np.ndarray]] = []
    flat_cells: list[np.ndarray] = []
    for idx in range(lattice_rows * lattice_cols):
        base = expected if idx == mirror_index else canonical
        min_flips = 1
        noisy = _add_noise(base, noise_palette, min_flips=min_flips, max_flips=max_noise)
        flat_cells.append(noisy)

    try:
        majority = _majority_patch(flat_cells)
    except AssertionError:
        return None
    if not np.array_equal(majority, canonical):
        return None

    scores = _mirror_scores(flat_cells, canonical)
    if sum(score > 0 for score in scores) != 1:
        return None
    if int(np.argmax(scores)) != mirror_index:
        return None
    if np.array_equal(flat_cells[mirror_index], expected):
        return None

    for rr in range(lattice_rows):
        row_cells = []
        for cc in range(lattice_cols):
            row_cells.append(flat_cells[rr * lattice_cols + cc])
        noisy_rows.append(row_cells)

    grid = _build_grid(noisy_rows, line_color=line_color, line_thickness=line_thickness)
    if max(grid.shape) > 30:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-impostor input")
