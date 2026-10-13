from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
DIVIDER = 7
COLOR_POOL = (1, 2, 3, 4, 6, 8, 9)
CELL_POSITIONS = ((0, 0), (0, 1), (1, 0), (1, 1))
GENERATION_RETRIES = 400

_GEN_COUNTER = 0


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    comps: list[np.ndarray] = []

    for r0 in range(height):
        for c0 in range(width):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _ranges_between(size: int, dividers: list[int]) -> list[tuple[int, int]]:
    points = [-1] + list(dividers) + [int(size)]
    return [
        (points[i] + 1, points[i + 1] - 1)
        for i in range(len(points) - 1)
        if points[i] + 1 <= points[i + 1] - 1
    ]


def _find_divider(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    height, width = grid.shape
    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        rows = [r for r in range(height) if np.all(grid[r, :] == color)]
        cols = [c for c in range(width) if np.all(grid[:, c] == color)]
        if rows and cols:
            candidates.append((int(color), rows, cols))
    assert len(candidates) == 1, "expected exactly one divider color"
    return candidates[0]


def _reflect_to_missing(mask: np.ndarray, src: tuple[int, int], missing: tuple[int, int]) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    if src[0] != missing[0]:
        out = out[::-1, :]
    if src[1] != missing[1]:
        out = out[:, ::-1]
    return out


def _parse_cells(grid: np.ndarray) -> tuple[
    int,
    int,
    list[tuple[int, int]],
    list[tuple[int, int]],
    dict[tuple[int, int], dict[str, np.ndarray | int]],
    tuple[int, int],
]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    divider, divider_rows, divider_cols = _find_divider(g, bg)
    assert divider == DIVIDER, "expected orange divider lines"
    assert len(divider_rows) == 1 and len(divider_cols) == 1, "expected a single horizontal and vertical divider"

    row_ranges = _ranges_between(g.shape[0], divider_rows)
    col_ranges = _ranges_between(g.shape[1], divider_cols)
    assert len(row_ranges) == 2 and len(col_ranges) == 2, "expected a 2x2 split"

    row_heights = {r1 - r0 + 1 for r0, r1 in row_ranges}
    col_widths = {c1 - c0 + 1 for c0, c1 in col_ranges}
    assert len(row_heights) == 1 and len(col_widths) == 1, "all cells must be equal-sized"

    occupied: dict[tuple[int, int], dict[str, np.ndarray | int]] = {}
    empty_cells: list[tuple[int, int]] = []
    object_colors: list[int] = []

    for cell_r, (r0, r1) in enumerate(row_ranges):
        for cell_c, (c0, c1) in enumerate(col_ranges):
            window = g[r0 : r1 + 1, c0 : c1 + 1]
            colors = set(map(int, np.unique(window))) - {bg, divider}
            assert divider not in colors, "divider color must not appear inside a cell"

            if not colors:
                assert np.all(window == bg), "the missing cell must be pure background"
                empty_cells.append((cell_r, cell_c))
                continue

            assert len(colors) == 1, "each occupied cell must contain exactly one object color"
            color = colors.pop()
            mask = window == color
            assert np.all(window[~mask] == bg), "cell background must stay black"
            comps = _components_mask(mask)
            assert len(comps) == 1, "each cell object must be 4-connected"
            pixel_count = int(mask.sum())
            assert 4 <= pixel_count <= 10, "object size must stay within the intended range"
            occupied[(cell_r, cell_c)] = {"color": int(color), "mask": mask}
            object_colors.append(int(color))

    assert len(empty_cells) == 1, "expected exactly one missing cell"
    assert len(occupied) == 3, "expected exactly three occupied cells"
    assert len(set(object_colors)) == 1, "all present objects must have the same color"
    return bg, divider, row_ranges, col_ranges, occupied, empty_cells[0]


def _infer_missing_mask(
    occupied: dict[tuple[int, int], dict[str, np.ndarray | int]],
    missing: tuple[int, int],
) -> np.ndarray:
    predictions = [
        _reflect_to_missing(info["mask"], src, missing)
        for src, info in sorted(occupied.items())
    ]
    first = predictions[0]
    assert all(np.array_equal(first, pred) for pred in predictions[1:]), "reflections must agree"
    assert 4 <= int(first.sum()) <= 10, "inferred object size out of range"
    assert len(_components_mask(first)) == 1, "inferred object must be 4-connected"
    return first


def _summarize_cells(
    grid: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
    bg: int,
    divider: int,
) -> np.ndarray:
    out = np.full((2, 2), bg, dtype=int)
    for cell_r, (r0, r1) in enumerate(row_ranges):
        for cell_c, (c0, c1) in enumerate(col_ranges):
            window = grid[r0 : r1 + 1, c0 : c1 + 1]
            colors = sorted(set(map(int, np.unique(window))) - {bg, divider})
            assert len(colors) <= 1, "each cell must summarize to at most one color"
            if colors:
                out[cell_r, cell_c] = int(colors[0])
    return out


def _is_nontrivial_base(mask: np.ndarray) -> bool:
    if len(_components_mask(mask)) != 1:
        return False
    pixel_count = int(mask.sum())
    if not (4 <= pixel_count <= 10):
        return False
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_coords(coords)
    if r1 == r0 or c1 == c0:
        return False
    if np.array_equal(mask, mask[::-1, :]):
        return False
    if np.array_equal(mask, mask[:, ::-1]):
        return False
    if np.array_equal(mask, mask[::-1, ::-1]):
        return False
    return True


def _grow_connected_patch(height: int, width: int, pixels: int) -> np.ndarray | None:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < pixels:
        frontier = []
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    patch = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        patch[r, c] = True
    return patch


def _generate_base_mask(cell_h: int, cell_w: int) -> np.ndarray:
    for _ in range(300):
        patch_h = random.randint(3, min(5, cell_h - 2))
        patch_w = random.randint(3, min(5, cell_w - 2))
        pixels = random.randint(4, min(10, patch_h * patch_w - 1))
        patch = _grow_connected_patch(patch_h, patch_w, pixels)
        if patch is None:
            continue

        full = np.zeros((cell_h, cell_w), dtype=bool)
        r0 = random.randint(1, cell_h - patch_h - 1)
        c0 = random.randint(1, cell_w - patch_w - 1)
        full[r0 : r0 + patch_h, c0 : c0 + patch_w] = patch
        if _is_nontrivial_base(full):
            return full
    raise RuntimeError("failed to build a nontrivial base mask")


def _expected_summary(color: int) -> np.ndarray:
    return np.full((2, 2), int(color), dtype=int)


def _validate_generated(
    grid: np.ndarray,
    expected_base: np.ndarray,
    expected_missing: tuple[int, int],
    expected_color: int,
) -> bool:
    try:
        bg, divider, row_ranges, col_ranges, occupied, missing = _parse_cells(grid)
        if missing != expected_missing:
            return False
        if any(int(info["color"]) != int(expected_color) for info in occupied.values()):
            return False

        inferred = _infer_missing_mask(occupied, missing)
        if not np.array_equal(inferred, expected_base):
            return False

        transformed_masks = [
            _reflect_to_missing(info["mask"], src, missing)
            for src, info in sorted(occupied.items())
        ]
        if len({m.tobytes() for m in transformed_masks}) != 1:
            return False

        present_masks = [np.asarray(info["mask"], dtype=bool) for info in occupied.values()]
        if len({m.tobytes() for m in present_masks}) != 3:
            return False

        naive = _summarize_cells(grid, row_ranges, col_ranges, bg, divider)
        expected = _expected_summary(expected_color)
        if np.array_equal(naive, expected):
            return False

        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False
    except Exception:
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the missing mirrored cell-object in a 2x2 divider table, then compress the completed table to one color per cell.

    Input structure:
    - Black is the background and orange forms one full horizontal divider row and one full vertical divider column.
    - The dividers split the grid into four equal-sized rectangular cells arranged as a 2x2 table.
    - Three cells each contain one 4-connected monochrome object on black, and the fourth cell is empty.
    - The three visible objects share one color and are horizontal/vertical mirror variants of the same cell-local shape.

    Rule:
    - Locate the empty cell and reflect each visible object into that cell's coordinate system according to its relative row and column position.
    - Require all three reflected predictions to match, then paint that inferred object into the empty cell using the shared color.
    - After the 2x2 table is complete, summarize each cell by its unique non-black, non-orange color.
    - Return the resulting 2x2 color table.

    Key insight:
    - The missing cell cannot be summarized correctly until the 2x2 mirror quartet is completed; direct compression would leave a false black summary in the empty slot.
    """
    g = np.asarray(grid, dtype=int)
    bg, divider, row_ranges, col_ranges, occupied, missing = _parse_cells(g)

    inferred = _infer_missing_mask(occupied, missing)
    color = int(next(iter(occupied.values()))["color"])

    filled = g.copy()
    r0, r1 = row_ranges[missing[0]]
    c0, c1 = col_ranges[missing[1]]
    target = filled[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(target == bg), "missing cell must be empty before completion"
    target[inferred] = color

    summary = _summarize_cells(filled, row_ranges, col_ranges, bg, divider)
    assert np.all(summary == color), "the completed quartet should summarize to one shared color"
    return summary


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    forced_missing = CELL_POSITIONS[_GEN_COUNTER % len(CELL_POSITIONS)]
    forced_color = COLOR_POOL[(_GEN_COUNTER // len(CELL_POSITIONS)) % len(COLOR_POOL)]

    for _ in range(GENERATION_RETRIES):
        cell_h = random.randint(7, 10)
        cell_w = random.randint(7, 10)
        grid = np.full((2 * cell_h + 1, 2 * cell_w + 1), BACKGROUND, dtype=int)
        grid[cell_h, :] = DIVIDER
        grid[:, cell_w] = DIVIDER

        base_mask = _generate_base_mask(cell_h, cell_w)
        row_ranges = [(0, cell_h - 1), (cell_h + 1, 2 * cell_h)]
        col_ranges = [(0, cell_w - 1), (cell_w + 1, 2 * cell_w)]

        for pos in CELL_POSITIONS:
            if pos == forced_missing:
                continue
            local_mask = _reflect_to_missing(base_mask, forced_missing, pos)
            r0, r1 = row_ranges[pos[0]]
            c0, c1 = col_ranges[pos[1]]
            grid[r0 : r1 + 1, c0 : c1 + 1][local_mask] = forced_color

        if _validate_generated(grid, base_mask, forced_missing, forced_color):
            _GEN_COUNTER += 1
            return grid

    raise RuntimeError("failed to generate a valid input")
