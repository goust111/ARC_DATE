from __future__ import annotations

import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_fill_holes

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, paste


INPUT_SHAPE_COLORS = (Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE)
GEN_MAX_ATTEMPTS = 400


def _separator_indices(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sep_rows = np.flatnonzero(np.all(np.isin(grid, (Color.BLUE, Color.RED)), axis=1))
    sep_cols = np.flatnonzero(np.all(np.isin(grid, (Color.BLUE, Color.RED)), axis=0))
    assert sep_rows.size >= 3 and sep_cols.size >= 3, "expected a separator lattice with border lines"
    return sep_rows, sep_cols


def _clean_grid(grid: np.ndarray, sep_rows: np.ndarray, sep_cols: np.ndarray) -> np.ndarray:
    cleaned = np.asarray(grid, dtype=int).copy()
    line_mask = np.zeros_like(cleaned, dtype=bool)
    line_mask[sep_rows, :] = True
    line_mask[:, sep_cols] = True
    cleaned[(cleaned == Color.RED) & line_mask] = Color.BLUE
    cleaned[(cleaned == Color.RED) & ~line_mask] = Color.BLACK
    return cleaned


def _has_hole(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    filled = binary_fill_holes(mask).astype(bool)
    return bool(np.any(filled & ~mask))


def _cell_bounds(sep_rows: np.ndarray, sep_cols: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    bounds: list[tuple[int, int, int, int, int, int]] = []
    for ri in range(len(sep_rows) - 1):
        for ci in range(len(sep_cols) - 1):
            r0 = int(sep_rows[ri] + 1)
            r1 = int(sep_rows[ri + 1] - 1)
            c0 = int(sep_cols[ci] + 1)
            c1 = int(sep_cols[ci + 1] - 1)
            bounds.append((ri, ci, r0, r1, c0, c1))
    return bounds


def _analyze_cell(cleaned: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> tuple[bool, int, tuple[tuple[int, int], ...]]:
    cell = cleaned[r0 : r1 + 1, c0 : c1 + 1]
    comps = find_components(cell, bg=Color.BLACK, connectivity=4, monochromatic=False)
    assert len(comps) == 1, "each cell must contain exactly one 4-connected shape"
    coords = comps[0]["coords"]
    colors = sorted(set(map(int, cell[tuple(coords.T)])))
    assert len(colors) == 1 and colors[0] in INPUT_SHAPE_COLORS, "unexpected cell shape color"

    shifted = coords - coords.min(axis=0)
    signature = tuple(sorted((int(r), int(c)) for r, c in shifted))
    mask = np.zeros((int(coords[:, 0].max()) + 1, int(coords[:, 1].max()) + 1), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return _has_hole(mask), colors[0], signature


def _intersection_color(statuses: np.ndarray, ri: int, ci: int) -> int:
    neighbors: list[bool] = []
    for dr in (-1, 0):
        for dc in (-1, 0):
            rr = ri + dr
            cc = ci + dc
            if 0 <= rr < statuses.shape[0] and 0 <= cc < statuses.shape[1]:
                neighbors.append(bool(statuses[rr, cc]))
    assert neighbors, "every intersection must touch at least one cell"
    if all(neighbors):
        return Color.GREEN
    if not any(neighbors):
        return Color.YELLOW
    return Color.BLUE


def main(grid: np.ndarray) -> np.ndarray:
    """Classify each lattice cell by whether its shape has a hole, then send that topology back to the lattice intersections.

    Input structure:
    - A black background contains full blue separator rows and columns that include the outer border, with some separator pixels overwritten by red noise.
    - Every cell interior contains exactly one 4-connected colored shape, and some black interior pixels may also be replaced by red noise.
    - Shape colors are arbitrary distractors from a small palette; the relevant structure is whether the shape encloses a background hole after noise is removed.

    Rule:
    - Detect separator rows and columns as the full rows/columns containing only blue or red.
    - Restore red separator pixels to blue and erase red interior noise back to black.
    - In each cell, classify the surviving shape as hollow if its mask encloses a background hole, otherwise solid.
    - Recolor each separator intersection from the classifications of its adjacent cells: all hollow -> green, all solid -> yellow, mixed -> blue; all non-intersection separator pixels stay blue and cell shapes stay unchanged.

    Key insight:
    - The output signal is carried by the lattice, but it is determined by topology inside the cells: the only thing that matters is whether each cell's shape has a hole, not the shape's color, size, or position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    sep_rows, sep_cols = _separator_indices(g)
    cleaned = _clean_grid(g, sep_rows, sep_cols)

    statuses = np.zeros((len(sep_rows) - 1, len(sep_cols) - 1), dtype=bool)
    for ri, ci, r0, r1, c0, c1 in _cell_bounds(sep_rows, sep_cols):
        has_hole, _color, _sig = _analyze_cell(cleaned, r0, r1, c0, c1)
        statuses[ri, ci] = has_hole

    out = cleaned.copy()
    for ri, row in enumerate(sep_rows):
        for ci, col in enumerate(sep_cols):
            out[int(row), int(col)] = _intersection_color(statuses, ri, ci)
    return out


def _normalized_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    shifted = coords - coords.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _random_connected_mask(height: int, width: int, area: int, rng: np.random.Generator) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    seed = (int(rng.integers(0, height)), int(rng.integers(0, width)))
    mask[seed] = True

    while int(mask.sum()) < area:
        frontier = sorted(
            {
                (nr, nc)
                for r, c in np.argwhere(mask)
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]
            }
        )
        if not frontier:
            return None
        rr, cc = frontier[int(rng.integers(0, len(frontier)))]
        mask[rr, cc] = True
    return mask


def _random_solid_shape(height: int, width: int, used_sigs: set[tuple[tuple[int, int], ...]], rng: np.random.Generator) -> np.ndarray | None:
    max_area = min(8, height * width - 1)
    if max_area < 3:
        return None

    for _ in range(200):
        area = int(rng.integers(3, max_area + 1))
        mask = _random_connected_mask(height, width, area, rng)
        if mask is None or _has_hole(mask):
            continue
        coords = np.argwhere(mask)
        rspan = int(coords[:, 0].ptp()) + 1
        cspan = int(coords[:, 1].ptp()) + 1
        if rspan == 1 or cspan == 1:
            continue
        if int(mask.sum()) == rspan * cspan:
            continue
        sig = _normalized_signature(mask)
        if sig in used_sigs:
            continue
        used_sigs.add(sig)
        return mask
    return None


def _ring_shape(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    top = int(rng.integers(0, height - 2))
    left = int(rng.integers(0, width - 2))
    mask[top : top + 3, left] = True
    mask[top : top + 3, left + 2] = True
    mask[top, left : left + 3] = True
    mask[top + 2, left : left + 3] = True
    return mask


def _render_clean_grid(
    row_heights: list[int],
    col_widths: list[int],
    statuses: np.ndarray,
    colors: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[tuple[int, int], ...]]]:
    height = sum(row_heights) + len(row_heights) + 1
    width = sum(col_widths) + len(col_widths) + 1
    clean = np.full((height, width), Color.BLACK, dtype=int)

    sep_rows = [0]
    for cell_h in row_heights:
        sep_rows.append(sep_rows[-1] + cell_h + 1)
    sep_cols = [0]
    for cell_w in col_widths:
        sep_cols.append(sep_cols[-1] + cell_w + 1)
    sep_rows_arr = np.array(sep_rows, dtype=int)
    sep_cols_arr = np.array(sep_cols, dtype=int)

    clean[sep_rows_arr, :] = Color.BLUE
    clean[:, sep_cols_arr] = Color.BLUE

    used_sigs: set[tuple[tuple[int, int], ...]] = set()
    placed_sigs: list[tuple[tuple[int, int], ...]] = []
    for ri, ci, r0, r1, c0, c1 in _cell_bounds(sep_rows_arr, sep_cols_arr):
        cell_h = r1 - r0 + 1
        cell_w = c1 - c0 + 1
        if statuses[ri, ci]:
            mask = _ring_shape(cell_h, cell_w, rng)
            sig = _normalized_signature(mask)
            used_sigs.add(sig)
        else:
            mask = _random_solid_shape(cell_h, cell_w, used_sigs, rng)
            if mask is None:
                raise ValueError("failed to sample a unique solid shape")
            sig = _normalized_signature(mask)

        patch = np.zeros((cell_h, cell_w), dtype=int)
        patch[mask] = int(colors[ri, ci])
        paste(clean, patch, r0, c0, transparent=Color.BLACK)
        placed_sigs.append(sig)
    return clean, sep_rows_arr, sep_cols_arr, placed_sigs


def _planned_output(clean: np.ndarray, sep_rows: np.ndarray, sep_cols: np.ndarray, statuses: np.ndarray) -> np.ndarray:
    out = clean.copy()
    for ri, row in enumerate(sep_rows):
        for ci, col in enumerate(sep_cols):
            out[int(row), int(col)] = _intersection_color(statuses, ri, ci)
    return out


def _add_noise(
    clean: np.ndarray,
    sep_rows: np.ndarray,
    sep_cols: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, int]:
    inp = clean.copy()
    line_mask = np.zeros_like(inp, dtype=bool)
    line_mask[sep_rows, :] = True
    line_mask[:, sep_cols] = True

    line_cells = np.argwhere(line_mask & (inp == Color.BLUE))
    rng.shuffle(line_cells)
    line_noise_count = int(rng.integers(1, max(2, min(10, len(line_cells)) + 1)))
    for r, c in line_cells[:line_noise_count]:
        inp[int(r), int(c)] = Color.RED

    interior_black = np.argwhere(~line_mask & (inp == Color.BLACK))
    rng.shuffle(interior_black)
    max_interior = min(len(interior_black), 8)
    assert max_interior >= 1, "need available interior background for red noise"
    interior_noise_count = int(rng.integers(1, max_interior + 1))
    for r, c in interior_black[:interior_noise_count]:
        inp[int(r), int(c)] = Color.RED

    return inp, line_noise_count, interior_noise_count


def _count_intersection_classes(out: np.ndarray, sep_rows: np.ndarray, sep_cols: np.ndarray) -> dict[int, int]:
    values = out[np.ix_(sep_rows, sep_cols)].ravel()
    return {color: int(np.count_nonzero(values == color)) for color in (Color.GREEN, Color.YELLOW, Color.BLUE)}


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    planned_rows: np.ndarray,
    planned_cols: np.ndarray,
    statuses: np.ndarray,
    colors: np.ndarray,
    placed_sigs: list[tuple[tuple[int, int], ...]],
    line_noise_count: int,
    interior_noise_count: int,
) -> None:
    sep_rows, sep_cols = _separator_indices(inp)
    assert np.array_equal(sep_rows, planned_rows)
    assert np.array_equal(sep_cols, planned_cols)
    assert line_noise_count > 0 and interior_noise_count > 0

    cleaned = _clean_grid(inp, sep_rows, sep_cols)
    seen_by_color: dict[int, set[bool]] = {}
    seen_sigs: set[tuple[tuple[int, int], ...]] = set()
    idx = 0
    for ri, ci, r0, r1, c0, c1 in _cell_bounds(sep_rows, sep_cols):
        has_hole, color, sig = _analyze_cell(cleaned, r0, r1, c0, c1)
        assert has_hole == bool(statuses[ri, ci])
        assert color == int(colors[ri, ci])
        assert sig == placed_sigs[idx]
        seen_by_color.setdefault(color, set()).add(bool(has_hole))
        assert sig not in seen_sigs
        seen_sigs.add(sig)
        idx += 1

    class_counts = _count_intersection_classes(expected, sep_rows, sep_cols)
    assert class_counts[Color.GREEN] > 0
    assert class_counts[Color.YELLOW] > 0
    assert class_counts[Color.BLUE] > 0
    assert any(vals == {False, True} for vals in seen_by_color.values())
    assert np.any(inp != expected)
    solved = main(inp)
    assert np.array_equal(solved, expected)


def _choose_layout(rng: np.random.Generator) -> tuple[list[int], list[int]]:
    row_count = int(rng.integers(2, 4))
    col_count = int(rng.integers(2, 4))
    row_heights = [int(rng.integers(3, 6)) for _ in range(row_count)]
    col_widths = [int(rng.integers(3, 6)) for _ in range(col_count)]
    return row_heights, col_widths


def _corner_choices(shape: tuple[int, int]) -> list[tuple[int, int]]:
    rows, cols = shape
    return [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]


def _color_grid(statuses: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    colors = np.empty(statuses.shape, dtype=int)
    shared = int(rng.choice(INPUT_SHAPE_COLORS))
    hole_pos = tuple(map(int, np.argwhere(statuses)[0]))
    solid_positions = [tuple(map(int, p)) for p in np.argwhere(~statuses)]
    shared_solid = solid_positions[int(rng.integers(0, len(solid_positions)))]

    for ri in range(statuses.shape[0]):
        for ci in range(statuses.shape[1]):
            if (ri, ci) == hole_pos or (ri, ci) == shared_solid:
                colors[ri, ci] = shared
            else:
                colors[ri, ci] = int(rng.choice(INPUT_SHAPE_COLORS))
    return colors


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    row_heights, col_widths = _choose_layout(rng)
    statuses = np.zeros((len(row_heights), len(col_widths)), dtype=bool)
    hole_r, hole_c = _corner_choices(statuses.shape)[int(rng.integers(0, 4))]
    statuses[hole_r, hole_c] = True
    colors = _color_grid(statuses, rng)

    try:
        clean, sep_rows, sep_cols, placed_sigs = _render_clean_grid(row_heights, col_widths, statuses, colors, rng)
    except ValueError:
        return None

    expected = _planned_output(clean, sep_rows, sep_cols, statuses)
    inp, line_noise_count, interior_noise_count = _add_noise(clean, sep_rows, sep_cols, rng)

    try:
        _semantic_checks(
            inp,
            expected,
            sep_rows,
            sep_cols,
            statuses,
            colors,
            placed_sigs,
            line_noise_count,
            interior_noise_count,
        )
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        sample = _attempt_generate(rng)
        if sample is not None:
            return sample
    raise RuntimeError(f"failed to generate a valid sample after {GEN_MAX_ATTEMPTS} attempts")


if __name__ == "__main__":
    p = Path(__file__)
    mod = SourceFileLoader("arc_task_self", str(p)).load_module()
    for _ in range(3):
        x = mod.generate_input()
        y = mod.main(x)
        print(x.shape, y.shape, np.unique(y, return_counts=True))
