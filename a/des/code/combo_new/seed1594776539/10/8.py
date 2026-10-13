import random

import numpy as np


BG = 0
OBJECT_COLOR = 5
PALETTE_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_SIZE_BOUNDS = (15, 20)
PALETTE_LEN_BOUNDS = (7, 8)
OBJECT_COUNT_BOUNDS = (4, 6)
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 40
PLACE_PADDING = 1


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append(np.array(coords, dtype=int))

    return components


def _palette_from_row(row: np.ndarray) -> list[int]:
    palette: list[int] = []
    for value in map(int, row):
        if value == BG:
            break
        palette.append(value)
    assert palette, "top row must start with a non-empty palette"
    assert len(set(palette)) == len(palette), "palette entries must be distinct"
    if len(palette) < len(row):
        assert bool(np.all(row[len(palette) :] == BG)), "palette row must have black suffix"
    return palette


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each scattered object by using its area as an index into the top-row palette.

    Input structure:
    - The background is black and the grid is at most 30×30.
    - Row 0 begins with a contiguous palette of distinct non-black colors; any remaining cells in that row are black.
    - Rows 1+ contain several non-overlapping 4-connected monochrome foreground objects on the black background.

    Rule:
    - Read the contiguous non-black prefix of row 0 as the palette.
    - For each 4-connected foreground object below the palette row, count its pixels to get area N.
    - Repaint that entire object with the color at palette column N-1, while leaving the palette row and black background unchanged.

    Key insight:
    - The top row is a lookup table, and each object's hidden indexing feature is its area rather than its original color or shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    palette = _palette_from_row(g[0])
    out = g.copy()

    for comp in _component_coords(g[1:, :] != BG):
        area = int(comp.shape[0])
        assert 1 <= area <= len(palette), "object area must fit inside palette length"
        rows = comp[:, 0] + 1
        cols = comp[:, 1]
        out[rows, cols] = int(palette[area - 1])

    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    norm = _normalize_cells(cells)
    height = max(r for r, _ in norm) + 1
    width = max(c for _, c in norm) + 1
    mask = np.zeros((height, width), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nbr in _neighbors4(r, c):
            if nbr in cells and nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return len(seen) == len(cells)


def _blob_shape(area: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = sorted(
            {
                nbr
                for r, c in cells
                for nbr in _neighbors4(r, c)
                if nbr not in cells
            }
        )
        cells.add(random.choice(frontier))
    return _normalize_cells(cells)


def _path_shape(area: int) -> set[tuple[int, int]] | None:
    for _ in range(SHAPE_ATTEMPTS):
        ordered = [(0, 0)]
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = [nbr for nbr in _neighbors4(*ordered[-1]) if nbr not in cells]
            if not frontier:
                break
            nxt = random.choice(frontier)
            ordered.append(nxt)
            cells.add(nxt)
        if len(cells) == area:
            return _normalize_cells(cells)
    return None


def _carved_shape(area: int) -> set[tuple[int, int]] | None:
    options = [
        (h, w)
        for h in range(1, area + 1)
        for w in range(1, area + 1)
        if area <= h * w <= area + 4
    ]
    random.shuffle(options)

    for h, w in options[: max(1, min(len(options), SHAPE_ATTEMPTS))]:
        cells = {(r, c) for r in range(h) for c in range(w)}
        while len(cells) > area:
            boundary = [
                cell
                for cell in cells
                if any(nbr not in cells for nbr in _neighbors4(*cell))
            ]
            removable = [
                cell
                for cell in boundary
                if len(cells) > 1 and _is_connected(cells - {cell})
            ]
            if not removable:
                break
            cells.remove(random.choice(removable))
        if len(cells) == area and _is_connected(cells):
            return _normalize_cells(cells)
    return None


def _random_shape(area: int) -> np.ndarray:
    if area == 1:
        return np.array([[True]], dtype=bool)

    builders = [_blob_shape, _path_shape, _carved_shape]
    for _ in range(SHAPE_ATTEMPTS):
        builder = random.choice(builders)
        cells = builder(area)
        if cells is None:
            continue
        mask = _cells_to_mask(cells)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, k=random.randint(0, 3))
        mask = _trim_mask(mask)
        if int(mask.sum()) == area:
            return mask

    return _trim_mask(_cells_to_mask(_blob_shape(area)))


def _candidate_positions(blocked: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = blocked.shape
    sh, sw = mask.shape
    positions: list[tuple[int, int]] = []

    for r0 in range(2, h - sh):
        for c0 in range(1, w - sw):
            r1 = r0 + sh
            c1 = c0 + sw
            if blocked[
                max(0, r0 - PLACE_PADDING) : min(h, r1 + PLACE_PADDING),
                max(0, c0 - PLACE_PADDING) : min(w, c1 + PLACE_PADDING),
            ].any():
                continue
            positions.append((r0, c0))

    random.shuffle(positions)
    return positions


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    view = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    view[mask] = int(color)


def _expected_output(grid: np.ndarray) -> np.ndarray:
    palette = _palette_from_row(grid[0])
    out = grid.copy()
    for comp in _component_coords(grid[1:, :] == OBJECT_COLOR):
        area = int(comp.shape[0])
        rows = comp[:, 0] + 1
        cols = comp[:, 1]
        out[rows, cols] = int(palette[area - 1])
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIZE_BOUNDS)
    width = random.randint(*GRID_SIZE_BOUNDS)
    palette_len = random.randint(*PALETTE_LEN_BOUNDS)
    object_count = random.randint(OBJECT_COUNT_BOUNDS[0], min(OBJECT_COUNT_BOUNDS[1], palette_len))

    palette = random.sample(PALETTE_POOL, palette_len)
    areas = random.sample(range(1, palette_len + 1), object_count)
    shapes = {area: _random_shape(area) for area in areas}

    grid = np.full((height, width), BG, dtype=int)
    grid[0, :palette_len] = np.array(palette, dtype=int)

    blocked = np.zeros((height, width), dtype=bool)
    blocked[0, :] = True

    placements: list[tuple[int, int, int, np.ndarray]] = []
    for area in sorted(areas, reverse=True):
        mask = shapes[area]
        positions = _candidate_positions(blocked, mask)
        if not positions:
            return None
        top, left = positions[0]
        _paint_mask(grid, mask, top, left, OBJECT_COLOR)
        blocked[
            max(0, top - PLACE_PADDING) : min(height, top + mask.shape[0] + PLACE_PADDING),
            max(0, left - PLACE_PADDING) : min(width, left + mask.shape[1] + PLACE_PADDING),
        ] = True
        placements.append((area, top, left, mask))

    comps = _component_coords(grid[1:, :] == OBJECT_COLOR)
    if len(comps) != object_count:
        return None
    if sorted(int(comp.shape[0]) for comp in comps) != sorted(areas):
        return None
    if not np.all(grid[1:, :][grid[1:, :] != BG] == OBJECT_COLOR):
        return None

    expected = grid.copy()
    for area, top, left, mask in placements:
        _paint_mask(expected, mask, top, left, palette[area - 1])

    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(expected, actual):
        return None
    if np.array_equal(grid, actual):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
