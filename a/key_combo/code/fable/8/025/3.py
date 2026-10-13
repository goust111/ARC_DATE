from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import Color, colors_present, find_components, neighbors4


GENERATION_ATTEMPTS = 400
LAYOUTS = ((2, 2), (2, 3), (3, 2), (2, 4), (4, 2))
PALETTE = tuple(range(1, 10))
CELL_SIZE_MIN = 4
CELL_SIZE_MAX = 7


def _divider_info(grid: np.ndarray) -> dict[str, object]:
    H, W = grid.shape
    candidates: list[dict[str, object]] = []

    for color in colors_present(grid):
        full_rows = [r for r in range(H) if np.all(grid[r, :] == color)]
        full_cols = [c for c in range(W) if np.all(grid[:, c] == color)]
        if len(full_rows) < 3 or len(full_cols) < 3:
            continue
        if full_rows[0] != 0 or full_rows[-1] != H - 1:
            continue
        if full_cols[0] != 0 or full_cols[-1] != W - 1:
            continue

        row_steps = np.diff(full_rows)
        col_steps = np.diff(full_cols)
        if len(set(map(int, row_steps))) != 1 or len(set(map(int, col_steps))) != 1:
            continue
        if int(row_steps[0]) <= 1 or int(col_steps[0]) <= 1:
            continue

        candidates.append(
            {
                "color": int(color),
                "rows": full_rows,
                "cols": full_cols,
                "cell_h": int(row_steps[0] - 1),
                "cell_w": int(col_steps[0] - 1),
            }
        )

    assert len(candidates) == 1, "expected a unique divider-grid color"
    return candidates[0]


def _infer_background(grid: np.ndarray, divider: int) -> int:
    vals, counts = np.unique(grid[grid != divider], return_counts=True)
    assert len(vals) > 0, "expected non-divider cells"
    return int(vals[np.argmax(counts)])


def _extract_cells(grid: np.ndarray) -> tuple[int, int, list[dict[str, object]], tuple[int, int]]:
    info = _divider_info(grid)
    divider = int(info["color"])
    bg = _infer_background(grid, divider)
    rows = list(info["rows"])
    cols = list(info["cols"])
    cell_shape = (int(info["cell_h"]), int(info["cell_w"]))
    cells: list[dict[str, object]] = []

    for r_idx in range(len(rows) - 1):
        for c_idx in range(len(cols) - 1):
            r0, r1 = rows[r_idx] + 1, rows[r_idx + 1]
            c0, c1 = cols[c_idx] + 1, cols[c_idx + 1]
            patch = grid[r0:r1, c0:c1]
            assert patch.shape == cell_shape

            fg_colors = colors_present(patch, exclude=[bg, divider])
            assert len(fg_colors) == 1, "each cell must contain exactly one mask color"
            color = int(fg_colors[0])
            assert set(map(int, np.unique(patch))).issubset({bg, color})

            mask = patch == color
            assert mask.any(), "each cell must contain a non-empty mask"
            cells.append({"slice": (r0, r1, c0, c1), "color": color, "mask": mask})

    assert cells, "expected at least one cell"
    return bg, divider, cells, cell_shape


def main(grid: np.ndarray) -> np.ndarray:
    """Rebuild the shared template in every divider-grid cell by strict majority vote across damaged copies.

    Input structure:
    - One color forms complete horizontal and vertical divider lines that partition the canvas into equal-sized cells.
    - Inside each cell, exactly one other color marks a single-color mask on a shared background.
    - The masks are aligned damaged copies of one hidden template: true template pixels survive in a strict majority of cells, while spurious pixels appear in at most a minority.

    Rule:
    - Detect the divider grid and read every cell in local cell coordinates.
    - Vote independently at each local coordinate, keeping the pixels present in strictly more than half of the cells as the consensus template.
    - Clear each cell interior and stamp that consensus template back using the cell's original color, leaving divider lines unchanged.

    Key insight:
    - No individual cell is reliable: intersections lose genuine pixels and unions keep noise, so only aligned cross-copy strict-majority voting recovers the unique template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg, _divider, cells, cell_shape = _extract_cells(g)
    masks = np.stack([cell["mask"] for cell in cells]).astype(int)
    votes = masks.sum(axis=0)
    consensus = votes > (len(cells) // 2)
    assert consensus.shape == cell_shape
    assert consensus.any(), "consensus template must be non-empty"

    out = g.copy()
    for cell in cells:
        r0, r1, c0, c1 = cell["slice"]
        region = out[r0:r1, c0:c1]
        region[:] = bg
        region[consensus] = int(cell["color"])
    return out


def _sample_cell_size(cell_rows: int, cell_cols: int, cell_count: int, rng: random.Random) -> tuple[int, int]:
    for _ in range(100):
        h = rng.randint(CELL_SIZE_MIN, CELL_SIZE_MAX)
        w = rng.randint(CELL_SIZE_MIN, CELL_SIZE_MAX)
        if h * w < 2 * cell_count + 4:
            continue
        if cell_rows * h + cell_rows + 1 > 30:
            continue
        if cell_cols * w + cell_cols + 1 > 30:
            continue
        return h, w
    raise RuntimeError("failed to sample a valid cell size")


def _grow_connected_mask(h: int, w: int, area: int, rng: random.Random) -> np.ndarray:
    cells = {(rng.randrange(h), rng.randrange(w))}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        assert frontier, "connected growth frontier should not be empty"
        cells.add(rng.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    probe = np.zeros(mask.shape, dtype=int)
    probe[mask] = 1
    comps = find_components(probe, bg=0, connectivity=4, monochromatic=True, colors=[1])
    return len(comps) == 1


def _make_template(h: int, w: int, cell_count: int, rng: random.Random) -> np.ndarray:
    area = h * w
    min_fg = max(cell_count + 2, area // 3)
    max_fg = min(area - (cell_count + 2), (2 * area) // 3)
    assert min_fg <= max_fg

    for _ in range(120):
        target = rng.randint(min_fg, max_fg)
        mask = _grow_connected_mask(h, w, target, rng)
        if not _is_connected(mask):
            continue
        if np.all(mask) or not (~mask).any():
            continue
        if np.count_nonzero(mask.sum(axis=0)) < 2 or np.count_nonzero(mask.sum(axis=1)) < 2:
            continue
        return mask
    raise RuntimeError("failed to generate a template mask")


def _apply_damage(template: np.ndarray, cell_count: int, rng: random.Random) -> list[np.ndarray]:
    masks = [template.copy() for _ in range(cell_count)]
    fg_positions = [tuple(pos) for pos in np.argwhere(template)]
    bg_positions = [tuple(pos) for pos in np.argwhere(~template)]
    rng.shuffle(fg_positions)
    rng.shuffle(bg_positions)

    assert len(fg_positions) >= cell_count
    assert len(bg_positions) >= cell_count

    for idx in range(cell_count):
        r, c = fg_positions[idx]
        masks[idx][r, c] = False
    for idx in range(cell_count):
        r, c = bg_positions[idx]
        masks[idx][r, c] = True

    max_missing_per_pos = cell_count - (cell_count // 2 + 1)
    max_extra_per_pos = cell_count // 2

    for r, c in fg_positions[cell_count:]:
        if rng.random() < 0.20:
            count = rng.randint(1, max_missing_per_pos)
            for idx in rng.sample(range(cell_count), count):
                masks[idx][r, c] = False

    for r, c in bg_positions[cell_count:]:
        if rng.random() < 0.12:
            count = rng.randint(1, max_extra_per_pos)
            for idx in rng.sample(range(cell_count), count):
                masks[idx][r, c] = True

    return masks


def _render_grid(
    cell_rows: int,
    cell_cols: int,
    cell_h: int,
    cell_w: int,
    divider: int,
    colors: list[int],
    masks: list[np.ndarray],
) -> np.ndarray:
    H = cell_rows * cell_h + cell_rows + 1
    W = cell_cols * cell_w + cell_cols + 1
    grid = np.zeros((H, W), dtype=int)

    divider_rows = [r * (cell_h + 1) for r in range(cell_rows + 1)]
    divider_cols = [c * (cell_w + 1) for c in range(cell_cols + 1)]
    grid[divider_rows, :] = divider
    grid[:, divider_cols] = divider

    idx = 0
    for rr in range(cell_rows):
        for cc in range(cell_cols):
            r0 = divider_rows[rr] + 1
            c0 = divider_cols[cc] + 1
            color = colors[idx]
            region = grid[r0 : r0 + cell_h, c0 : c0 + cell_w]
            region[masks[idx]] = color
            idx += 1
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    cell_rows, cell_cols = rng.choice(LAYOUTS)
    cell_count = cell_rows * cell_cols
    cell_h, cell_w = _sample_cell_size(cell_rows, cell_cols, cell_count, rng)

    divider = rng.choice(PALETTE)
    cell_colors = list(PALETTE)
    cell_colors.remove(divider)
    rng.shuffle(cell_colors)
    cell_colors = cell_colors[:cell_count]

    template = _make_template(cell_h, cell_w, cell_count, rng)
    masks = _apply_damage(template, cell_count, rng)
    grid = _render_grid(cell_rows, cell_cols, cell_h, cell_w, divider, cell_colors, masks)

    try:
        bg, parsed_divider, cells, _shape = _extract_cells(grid)
        assert bg == Color.BLACK
        assert parsed_divider == divider

        consensus = np.sum(np.stack([cell["mask"] for cell in cells]).astype(int), axis=0) > (cell_count // 2)
        union = np.any(np.stack([cell["mask"] for cell in cells]), axis=0)
        inter = np.all(np.stack([cell["mask"] for cell in cells]), axis=0)

        assert np.array_equal(consensus, template)
        assert not np.array_equal(union, template)
        assert not np.array_equal(inter, template)

        for cell in cells:
            mask = cell["mask"]
            assert not np.array_equal(mask, template)
            assert np.any(template & ~mask)
            assert np.any(mask & ~template)

        counts = np.sum(np.stack(masks).astype(int), axis=0)
        assert np.all(counts[template] > cell_count // 2)
        assert np.all(counts[~template] <= cell_count // 2)

        return grid
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid majority-vote stamp input")
