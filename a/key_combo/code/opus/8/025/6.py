import itertools
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

GRID_BANDS = 3
DIVIDER_WIDTH = 1
MARKER_CELLS = [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)]
ROW_MARKER_COLORS = [BLUE, GREEN, YELLOW, PINK, ORANGE, TEAL]
MAX_GENERATION_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _segments(length: int, dividers: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(dividers) + [length]
    out = []
    for lo, hi in zip(bounds, bounds[1:]):
        out.append((lo + 1, hi))
    return out


def _find_divider_grid(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    for color in sorted(set(map(int, grid.ravel()))):
        divider_rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        divider_cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if len(divider_rows) != GRID_BANDS - 1 or len(divider_cols) != GRID_BANDS - 1:
            continue
        row_segments = _segments(h, divider_rows)
        col_segments = _segments(w, divider_cols)
        row_sizes = {r1 - r0 for r0, r1 in row_segments}
        col_sizes = {c1 - c0 for c0, c1 in col_segments}
        if len(row_segments) == GRID_BANDS and len(col_segments) == GRID_BANDS and row_sizes == col_sizes:
            return int(color), row_segments, col_segments
    raise AssertionError("divider grid not found")


def _cell(grid: np.ndarray, row_segments: list[tuple[int, int]], col_segments: list[tuple[int, int]], rr: int, cc: int) -> np.ndarray:
    r0, r1 = row_segments[rr]
    c0, c1 = col_segments[cc]
    return grid[r0:r1, c0:c1]


def _mask_coords(mask: np.ndarray) -> frozenset[tuple[int, int]]:
    return frozenset(map(tuple, np.argwhere(mask)))


def _d4_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for mirrored in (False, True):
        base = np.fliplr(mask) if mirrored else mask
        for k in range(4):
            variants.append(np.rot90(base, k))
    return variants


def _variant_coord_sets(mask: np.ndarray) -> list[frozenset[tuple[int, int]]]:
    return [_mask_coords(v) for v in _d4_variants(mask)]


def _paint_mask(cell: np.ndarray, coords: frozenset[tuple[int, int]], color: int) -> None:
    cell[:, :] = BG
    for r, c in coords:
        cell[r, c] = color


def _extract_template_and_variants(grid: np.ndarray) -> tuple[np.ndarray, list[frozenset[tuple[int, int]]], list[tuple[int, int]], list[tuple[int, int]], int]:
    divider_color, row_segments, col_segments = _find_divider_grid(grid)
    bg = get_bg(grid)
    template_cell = _cell(grid, row_segments, col_segments, 0, 0)
    template_colors = sorted(set(map(int, template_cell.ravel())) - {bg})
    assert template_colors == [RED]
    template_mask = template_cell == RED
    assert len(components_mask(template_mask)) == 1
    variants = _variant_coord_sets(template_mask)
    assert len(set(variants)) == 8
    assert divider_color == MAROON
    return template_mask, variants, row_segments, col_segments, bg


def _row_colors(grid: np.ndarray, row_segments: list[tuple[int, int]], col_segments: list[tuple[int, int]], bg: int) -> list[int]:
    colors = [RED]
    for rr in range(1, GRID_BANDS):
        left_cell = _cell(grid, row_segments, col_segments, rr, 0)
        present = sorted(set(map(int, left_cell.ravel())) - {bg})
        assert len(present) == 1
        colors.append(int(present[0]))
    return colors


def _resolve_marker(marker_coords: frozenset[tuple[int, int]], variants: list[frozenset[tuple[int, int]]]) -> int:
    matches = [idx for idx, coords in enumerate(variants) if marker_coords and marker_coords <= coords]
    assert len(matches) == 1
    return matches[0]


def _worker_cells() -> list[tuple[int, int]]:
    return list(MARKER_CELLS)


def _solve_metadata(grid: np.ndarray) -> dict:
    template_mask, variants, row_segments, col_segments, bg = _extract_template_and_variants(grid)
    row_colors = _row_colors(grid, row_segments, col_segments, bg)
    cell_size = template_mask.shape[0]
    info = {
        "template_mask": template_mask,
        "variants": variants,
        "row_segments": row_segments,
        "col_segments": col_segments,
        "row_colors": row_colors,
        "bg": bg,
        "cell_size": cell_size,
        "marker_map": {},
    }
    for rr, cc in _worker_cells():
        cell = _cell(grid, row_segments, col_segments, rr, cc)
        marker_coords = _mask_coords(cell == GRAY)
        assert 2 <= len(marker_coords) <= 3
        assert set(map(int, cell.ravel())) <= {bg, GRAY}
        info["marker_map"][(rr, cc)] = _resolve_marker(marker_coords, variants)
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """Replace each gray direction marker with the full rotated template colored by its row palette.

    Input structure
    - Maroon divider bars split the canvas into a 3x3 grid of equal square cells.
    - The top-left cell contains a red connected template shape, and the other two cells in the first column contain single-pixel row-color markers.
    - Each non-first-column cell contains only a small gray subset of some rigid transform of the template on a black background.

    Rule
    - Read the red template in the top-left cell and enumerate its 8 rigid transforms (4 rotations, each with and without horizontal reflection).
    - For every non-first-column cell, match its gray pixels to the unique transform whose template contains that subset.
    - Keep the first column unchanged, clear each matched cell to black, and stamp the full matched transform using the row color: red for row 0, and the left-column marker color for rows 1 and 2.

    Key insight
    - The gray pixels are not the object to copy; they are a uniquely identifying partial projection of one transformed template, while the row markers independently choose the stamp color.
    """
    g = np.asarray(grid, dtype=int)
    info = _solve_metadata(g)
    out = g.copy()
    for rr, cc in _worker_cells():
        cell = _cell(out, info["row_segments"], info["col_segments"], rr, cc)
        variant_idx = info["marker_map"][(rr, cc)]
        color = info["row_colors"][rr]
        _paint_mask(cell, info["variants"][variant_idx], color)
    return out


def _random_connected_shape(size: int, area: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    start = (random.randrange(size), random.randrange(size))
    mask[start] = True
    frontier = {start}
    while int(mask.sum()) < area:
        base_r, base_c = random.choice(tuple(frontier))
        nbrs = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = base_r + dr, base_c + dc
            if 0 <= nr < size and 0 <= nc < size and not mask[nr, nc]:
                nbrs.append((nr, nc))
        if nbrs:
            choice = random.choice(nbrs)
            mask[choice] = True
            frontier.add(choice)
        else:
            frontier.remove((base_r, base_c))
            if not frontier:
                frontier = set(map(tuple, np.argwhere(mask)))
    return mask


def _center_embed(mask: np.ndarray, cell_size: int) -> np.ndarray:
    work_size = mask.shape[0]
    out = np.zeros((cell_size, cell_size), dtype=bool)
    offset = (cell_size - work_size) // 2
    out[offset:offset + work_size, offset:offset + work_size] = mask
    return out


def _shape_ok(mask: np.ndarray) -> bool:
    area = int(mask.sum())
    if not (5 <= area <= 8):
        return False
    if len(components_mask(mask)) != 1:
        return False
    box = bbox(mask)
    if box is None:
        return False
    r0, c0, r1, c1 = box
    if (r1 - r0 + 1) < 2 or (c1 - c0 + 1) < 2:
        return False
    if (r1 - r0 + 1) + (c1 - c0 + 1) < 5:
        return False
    return True


def _marker_candidates(variants: list[frozenset[tuple[int, int]]]) -> dict[int, list[frozenset[tuple[int, int]]]]:
    all_candidates: dict[int, list[frozenset[tuple[int, int]]]] = {}
    for idx, coords in enumerate(variants):
        coords_list = sorted(coords)
        found: list[frozenset[tuple[int, int]]] = []
        for k in (2, 3):
            for subset in itertools.combinations(coords_list, k):
                sub = frozenset(subset)
                match_count = sum(sub <= other for other in variants)
                if match_count == 1:
                    found.append(sub)
        if found:
            random.shuffle(found)
            all_candidates[idx] = found
    return all_candidates


def _choose_template_and_markers(cell_size: int) -> tuple[np.ndarray, dict[int, frozenset[tuple[int, int]]]]:
    work_size = 5 if cell_size == 5 else 4
    for _ in range(200):
        area = random.randint(5, 8)
        raw = _random_connected_shape(work_size, area)
        if not _shape_ok(raw):
            continue
        template_mask = _center_embed(raw, cell_size)
        variants = _variant_coord_sets(template_mask)
        if len(set(variants)) != 8:
            continue
        candidates = _marker_candidates(variants)
        if len(candidates) < 6:
            continue
        chosen_ids = random.sample(sorted(candidates), 6)
        chosen = {idx: random.choice(candidates[idx]) for idx in chosen_ids}
        return template_mask, chosen
    raise RuntimeError("failed to build a valid asymmetric template")


def _build_grid(cell_size: int, template_mask: np.ndarray, chosen_markers: dict[int, frozenset[tuple[int, int]]]) -> np.ndarray:
    side = GRID_BANDS * cell_size + (GRID_BANDS - 1) * DIVIDER_WIDTH
    grid = np.full((side, side), BG, dtype=int)

    for cut in range(cell_size, side, cell_size + DIVIDER_WIDTH):
        grid[cut:cut + DIVIDER_WIDTH, :] = MAROON
        grid[:, cut:cut + DIVIDER_WIDTH] = MAROON

    for r, c in np.argwhere(template_mask):
        grid[r, c] = RED

    row_palette = [RED] + random.sample(ROW_MARKER_COLORS, 2)
    center = cell_size // 2
    for rr in range(1, GRID_BANDS):
        r0 = rr * (cell_size + DIVIDER_WIDTH)
        grid[r0 + center, center] = row_palette[rr]

    placements = _worker_cells()
    transform_ids = list(chosen_markers.keys())
    random.shuffle(transform_ids)
    for (rr, cc), variant_idx in zip(placements, transform_ids):
        r0 = rr * (cell_size + DIVIDER_WIDTH)
        c0 = cc * (cell_size + DIVIDER_WIDTH)
        for r, c in chosen_markers[variant_idx]:
            grid[r0 + r, c0 + c] = GRAY

    return grid


def _validate_instance(grid: np.ndarray) -> None:
    info = _solve_metadata(grid)
    assert info["bg"] == BG
    assert grid.shape[0] == grid.shape[1]
    assert info["cell_size"] in (5, 6)
    used = list(info["marker_map"].values())
    assert len(set(used)) == 6
    assert any(idx < 4 for idx in used)
    assert any(idx >= 4 for idx in used)

    out = main(grid)
    for rr, cc in _worker_cells():
        cell_in = _cell(grid, info["row_segments"], info["col_segments"], rr, cc)
        cell_out = _cell(out, info["row_segments"], info["col_segments"], rr, cc)
        assert np.all((cell_in == BG) | (cell_in == GRAY))
        expected = np.full_like(cell_out, BG)
        variant_idx = info["marker_map"][(rr, cc)]
        for r, c in info["variants"][variant_idx]:
            expected[r, c] = info["row_colors"][rr]
        assert np.array_equal(cell_out, expected)

    top_left_in = _cell(grid, info["row_segments"], info["col_segments"], 0, 0)
    top_left_out = _cell(out, info["row_segments"], info["col_segments"], 0, 0)
    assert np.array_equal(top_left_in, top_left_out)
    for rr in range(1, GRID_BANDS):
        left_in = _cell(grid, info["row_segments"], info["col_segments"], rr, 0)
        left_out = _cell(out, info["row_segments"], info["col_segments"], rr, 0)
        assert np.array_equal(left_in, left_out)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        cell_size = random.choice([5, 6])
        template_mask, chosen_markers = _choose_template_and_markers(cell_size)
        grid = _build_grid(cell_size, template_mask, chosen_markers)
        try:
            _validate_instance(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate valid input")
