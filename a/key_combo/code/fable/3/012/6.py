"""Self-excluding catalog cell lattice reconstruction.

A square lattice of n×n cells is encoded as one n×n input grid, where each
cell is itself an n×n pixel block. Every non-empty cell contains a single
color pattern, empty cells are the background, and one catalog cell is
identical to the global occupancy map except that it has its own position
explicitly erased.

Input structure:
- The grid is square with side H = W = n×n.
- Partition it into an n×n cell lattice, each cell being n×n pixels.
- Each cell is either empty (all background) or contains exactly one
  non-background color.
- Build the occupancy map O: a 1 at (r,c) iff cell (r,c) is non-empty.
- Exactly one non-empty cell has foreground-mask pattern that differs from O
  in exactly one pixel, that pixel being its own cell coordinate.
- No non-empty cell has pattern exactly equal to O.

Rule:
- Locate the unique non-empty cell whose mask differs from O at exactly one
  location and that location equals the cell’s own coordinate.
- Fill an n×n output grid with background, then paint cells corresponding to
  O’s ones using that catalog cell’s color.

Key insight:
- The special “catalog” cell is not missing a random pixel; it is missing
  precisely its own cell position in the occupancy map, which uniquely
  identifies both the target map and the output color.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import numpy as np


def _load_arc_utils_get_bg():
    """Load `arc_utils.get_bg` with a fallback search from this file location."""
    try:
        from arc_utils import get_bg  # type: ignore

        return get_bg
    except Exception:
        pass

    this_file = Path(__file__).resolve()
    for parent in (this_file, *this_file.parents):
        candidate = parent / "new" / "solvers" / "arc_utils.py"
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("_arc_utils_runtime", str(candidate))
            if spec is None or spec.loader is None:
                break
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.get_bg

    raise ModuleNotFoundError("arc_utils.get_bg not found")


get_bg = _load_arc_utils_get_bg()


def _extract_cells(g: np.ndarray):
    h, w = map(int, g.shape)
    assert h == w and h > 0, "input must be a non-empty square grid"

    n = int(np.sqrt(h))
    assert n * n == h, "grid size must be n^2 so that cells are n-by-n"

    bg = int(get_bg(g))
    occupancy = np.zeros((n, n), dtype=bool)
    cell_info: list[tuple[int, int, int | None, np.ndarray]] = []

    for cell_r in range(n):
        for cell_c in range(n):
            r0, c0 = cell_r * n, cell_c * n
            patch = g[r0 : r0 + n, c0 : c0 + n]

            fg_values = [int(v) for v in np.unique(patch) if int(v) != bg]
            if not fg_values:
                color = None
                mask = np.zeros((n, n), dtype=bool)
            else:
                assert len(fg_values) == 1, "each non-empty cell must contain one color"
                color = int(fg_values[0])
                mask = patch != bg
                assert np.any(mask), "non-empty cell must have at least one foreground pixel"
                occupancy[cell_r, cell_c] = True

            cell_info.append((cell_r, cell_c, color, mask))

    return occupancy, cell_info, bg, n


def _catalog_candidates(occupancy: np.ndarray, cell_info: list[tuple[int, int, int | None, np.ndarray]]) -> list[tuple[int, int, int]]:
    """Find non-empty cells whose mask differs from occupancy by exactly one self pixel."""
    hits: list[tuple[int, int, int]] = []

    for cell_r, cell_c, color, mask in cell_info:
        if color is None:
            continue

        diff = mask != occupancy
        if int(np.count_nonzero(diff)) != 1:
            continue

        (dr, dc), = np.argwhere(diff)
        if (int(dr), int(dc)) != (cell_r, cell_c):
            continue
        if not occupancy[dr, dc] or mask[dr, dc]:
            continue

        hits.append((cell_r, cell_c, color))

    return hits


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    occupancy, cell_info, bg, n = _extract_cells(g)

    non_empty_colors = [color for _r, _c, color, _mask in cell_info if color is not None]
    assert len(set(non_empty_colors)) == len(non_empty_colors), "cell colors should be unique"
    assert non_empty_colors, "expected at least one non-empty cell"

    hits = _catalog_candidates(occupancy, cell_info)
    assert len(hits) == 1, "expected exactly one catalog cell (self-excluding)"

    _cat_r, _cat_c, cat_color = hits[0]
    assert cat_color is not None

    out = np.full((n, n), int(bg), dtype=int)
    out[occupancy] = int(cat_color)
    return out


_BG_COLOR = 0
_FG_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
_CELL_COUNTS = (3, 4, 5)
_TRY_LIMIT = 2000


def _random_connected_mask(size: int, pixels: int) -> np.ndarray:
    n = int(size)
    pixels = int(pixels)
    if pixels <= 0:
        return np.zeros((n, n), dtype=bool)

    pixels = min(pixels, n * n)
    mask = np.zeros((n, n), dtype=bool)
    r, c = np.random.randint(0, n), np.random.randint(0, n)
    mask[r, c] = True

    frontier: list[tuple[int, int]] = []
    while int(mask.sum()) < pixels:
        frontier.clear()
        for rr in range(n):
            for cc in range(n):
                if not mask[rr, cc]:
                    continue
                if rr > 0 and not mask[rr - 1, cc]:
                    frontier.append((rr - 1, cc))
                if rr + 1 < n and not mask[rr + 1, cc]:
                    frontier.append((rr + 1, cc))
                if cc > 0 and not mask[rr, cc - 1]:
                    frontier.append((rr, cc - 1))
                if cc + 1 < n and not mask[rr, cc + 1]:
                    frontier.append((rr, cc + 1))

        if not frontier:
            break
        rr, cc = frontier[np.random.randint(0, len(frontier))]
        mask[rr, cc] = True

    return mask


def _random_cell_mask(
    size: int,
    occupancy: np.ndarray,
    self_cell: tuple[int, int],
    avoid_masks: list[np.ndarray] | None = None,
) -> np.ndarray:
    n = int(size)
    self_r, self_c = map(int, self_cell)
    avoid = [a.astype(bool) for a in avoid_masks] if avoid_masks else []

    for _ in range(200):
        if np.random.random() < 0.5:
            target = int(np.random.randint(1, n * n + 1))
            mask = _random_connected_mask(n, target)
        else:
            fill = float(np.random.uniform(0.18, 0.85))
            mask = np.random.random((n, n)) < fill
            if not mask.any():
                continue

        if not mask.any():
            continue

        # Each non-empty cell must be strictly non-empty.
        if int(mask.sum()) == 0:
            continue

        # Avoid exact occupancy pattern and other known excluded masks.
        if np.array_equal(mask, occupancy):
            continue
        if any(np.array_equal(mask, a) for a in avoid):
            continue

        diff = mask != occupancy
        if int(np.count_nonzero(diff)) == 1:
            (dr, dc), = np.argwhere(diff)
            if (int(dr), int(dc)) == (self_r, self_c) and occupancy[dr, dc] and not mask[dr, dc]:
                # This would form another valid self-excluding candidate.
                continue

        return mask

    raise RuntimeError("failed to generate a non-catalog cell mask")


def generate_input() -> np.ndarray:
    for _ in range(_TRY_LIMIT):
        n = int(np.random.choice(_CELL_COUNTS))
        grid_side = n * n

        if grid_side > 30:
            continue

        total_cells = n * n
        max_non_empty = min(total_cells - 1, len(_FG_COLOR_POOL))
        if max_non_empty < 2:
            continue

        # Key-branch: sparse vs dense occupancy.
        if np.random.random() < 0.5:
            non_empty_count = int(np.random.randint(2, max_non_empty + 1))
        else:
            dense_min = max(2, int(total_cells * 0.55))
            dense_max = max_non_empty
            if dense_min > dense_max:
                dense_min = 2
            non_empty_count = int(np.random.randint(dense_min, dense_max + 1))

        occupancy = np.zeros((n, n), dtype=bool)
        flat_positions = np.arange(total_cells)
        chosen = np.random.choice(flat_positions, size=non_empty_count, replace=False)
        occupancy.flat[chosen] = True

        catalog_flat = int(np.random.choice(chosen))
        catalog_cell = divmod(catalog_flat, n)

        colors = np.random.choice(_FG_COLOR_POOL, size=non_empty_count, replace=False)

        # Render lattice cells.
        grid = np.full((grid_side, grid_side), int(_BG_COLOR), dtype=int)
        non_empty_iter = iter(range(total_cells))

        catalog_mask = occupancy.copy()
        catalog_mask[catalog_cell] = False

        avoid_masks = [catalog_mask]
        color_map = {}

        flat_idx = 0
        cell_color_cache: dict[int, int] = {}
        for idx, c in enumerate(colors):
            color_map[int(chosen[idx])] = int(c)

        for r in range(n):
            for c in range(n):
                if (r * n + c) not in color_map:
                    continue

                color = int(color_map[r * n + c])
                if (r, c) == catalog_cell:
                    mask = catalog_mask
                else:
                    mask = _random_cell_mask(n, occupancy, (r, c), avoid_masks=avoid_masks)

                # Keep the lattice structure with each color appearing inside
                # exactly its own n×n cell.
                block = np.full((n, n), int(_BG_COLOR), dtype=int)
                block[mask] = color
                grid[r * n : (r + 1) * n, c * n : (c + 1) * n] = block

                avoid_masks.append(mask)
                flat_idx = flat_idx + 1

        try:
            actual = main(grid)
        except Exception:
            continue

        expected = np.full((n, n), int(_BG_COLOR), dtype=int)
        expected[occupancy] = int(color_map[catalog_flat])
        if np.array_equal(actual, expected):
            return grid

    raise RuntimeError("failed to generate a valid input for puzzle 6")
