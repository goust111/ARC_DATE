import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PALETTE_COLORS = (BLUE, RED, GREEN, YELLOW)
SPRITE_COLOR = 8
CORNERS = ((0, 0), (0, 2), (2, 2), (2, 0))
GRID_SIZE = 11
CELL_SIZE = 3
DIVIDERS = (3, 7)
MAX_GENERATION_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    if box is None:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_signature(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    cropped = _crop_mask(mask)
    return (
        int(cropped.shape[0]),
        int(cropped.shape[1]),
        tuple(int(v) for v in cropped.astype(int).ravel()),
    )


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(int(v) for v in coords[0])
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and bool(mask[rr, cc]):
                nxt = (rr, cc)
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return len(seen) == int(mask.sum())


def _canonical_rotation_family(mask: np.ndarray) -> list[np.ndarray]:
    variants = [np.rot90(mask, -k) for k in range(4)]
    signatures = [_mask_signature(variant) for variant in variants]
    if len(set(signatures)) != 4:
        raise AssertionError("rotation family must have four distinct members")
    canonical = variants[min(range(4), key=signatures.__getitem__)]
    return [np.rot90(canonical, -k) for k in range(4)]


def _split_bands(length: int, dividers: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for divider in sorted(dividers):
        if divider > start:
            bands.append((start, divider))
        start = divider + 1
    if start < length:
        bands.append((start, length))
    return bands


def _cell_bands(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    divider_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    divider_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    row_bands = _split_bands(grid.shape[0], divider_rows)
    col_bands = _split_bands(grid.shape[1], divider_cols)
    assert len(row_bands) == 3 and len(col_bands) == 3, "expected a 3x3 divider grid"
    assert all(r1 - r0 == CELL_SIZE for r0, r1 in row_bands), "each cell must be 3 rows high"
    assert all(c1 - c0 == CELL_SIZE for c0, c1 in col_bands), "each cell must be 3 cols wide"
    return row_bands, col_bands


def _extract_cells(grid: np.ndarray) -> list[dict]:
    row_bands, col_bands = _cell_bands(grid)
    cells: list[dict] = []
    for rr, (r0, r1) in enumerate(row_bands):
        for cc, (c0, c1) in enumerate(col_bands):
            cells.append(
                {
                    "grid_pos": (rr, cc),
                    "slice": (r0, r1, c0, c1),
                    "cell": grid[r0:r1, c0:c1].copy(),
                }
            )
    return cells


def _parse_task(grid: np.ndarray) -> tuple[list[dict], dict, list[np.ndarray], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    cells = _extract_cells(g)

    key_cell: dict | None = None
    non_key_cells: list[dict] = []
    for cell_info in cells:
        cell = cell_info["cell"]
        mask = (cell != BLACK) & (cell != GREY)
        count = int(mask.sum())
        if count == 4:
            corner_mask = np.zeros((CELL_SIZE, CELL_SIZE), dtype=bool)
            for r, c in CORNERS:
                corner_mask[r, c] = True
            assert np.array_equal(mask, corner_mask), "key cell must use exactly the four corners"
            corner_colors = [int(cell[r, c]) for r, c in CORNERS]
            assert len(set(corner_colors)) == 4, "key colors must be distinct"
            assert set(corner_colors) == set(PALETTE_COLORS), "key colors must be blue/red/green/yellow"
            assert key_cell is None, "key cell must be unique"
            key_cell = cell_info
        else:
            assert count >= 5, "non-key sprite must use at least five colored pixels"
            colors = set(int(v) for v in cell[mask])
            assert len(colors) == 1, "each non-key cell must contain one monochrome sprite"
            assert _is_connected(mask), "sprite must be 4-connected"
            non_key_cells.append(cell_info)

    assert key_cell is not None, "expected exactly one key cell"
    assert len(non_key_cells) == 8, "expected eight non-key cells"

    family = _canonical_rotation_family((non_key_cells[0]["cell"] != BLACK) & (non_key_cells[0]["cell"] != GREY))
    family_signatures = [_mask_signature(member) for member in family]
    orientations: list[tuple[int, int]] = []
    for cell_info in non_key_cells:
        mask = (cell_info["cell"] != BLACK) & (cell_info["cell"] != GREY)
        signature = _mask_signature(mask)
        assert signature in family_signatures, "all sprites must belong to one rotation family"
        orientation = family_signatures.index(signature)
        cell_info["orientation"] = orientation
        orientations.append((cell_info["grid_pos"][0] * 3 + cell_info["grid_pos"][1], orientation))
    return cells, key_cell, family, orientations


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each sprite cell by using its rotation variant as an index into the key cell's sparse corner palette.

    Input structure:
    - The grid is partitioned by one-pixel grey divider lines into a 3x3 board of nine 3x3 cells on a black background.
    - Exactly one cell is a key cell containing four colored corner pixels and no sprite; the other eight cells each contain one monochrome 4-connected sprite from a single asymmetric rotation family.
    - The key cell's colored corners are the four distinct colors blue, red, green, and yellow.

    Rule:
    - Split the grid into its nine cells and identify the unique key cell by its four non-black, non-grey corner pixels.
    - Read the key palette in clockwise corner order: top-left gives the 0 degree color, top-right the 90 degree color, bottom-right the 180 degree color, and bottom-left the 270 degree color.
    - Canonicalize the shared sprite family to a unique base orientation, determine which clockwise quarter-turn variant appears in each non-key cell, and recolor that entire sprite with the keyed color for the matching variant.
    - Leave the key cell, grey dividers, and black background unchanged.

    Key insight:
    - The four colored key pixels are a sparse lookup table: they only become meaningful after identifying each sprite's rotation variant within its shared asymmetric family.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    cells, key_cell, _family, _orientations = _parse_task(g)
    palette = {
        index: int(key_cell["cell"][r, c])
        for index, (r, c) in enumerate(CORNERS)
    }

    for cell_info in cells:
        if cell_info is key_cell:
            continue
        cell = cell_info["cell"]
        mask = (cell != BLACK) & (cell != GREY)
        color = palette[int(cell_info["orientation"])]
        r0, r1, c0, c1 = cell_info["slice"]
        patch = out[r0:r1, c0:c1]
        patch[mask] = color
    return out


def _enumerate_base_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1 << 9):
        area = bits.bit_count()
        if area < 5 or area > 6:
            continue
        mask = np.zeros((CELL_SIZE, CELL_SIZE), dtype=bool)
        for idx in range(9):
            if (bits >> idx) & 1:
                mask[idx // CELL_SIZE, idx % CELL_SIZE] = True
        if not _is_connected(mask):
            continue
        try:
            family = _canonical_rotation_family(mask)
        except AssertionError:
            continue
        if len({_mask_signature(member) for member in family}) == 4:
            masks.append(mask)
    assert masks, "expected at least one asymmetric 3x3 sprite mask"
    return masks


BASE_MASKS = _enumerate_base_masks()


def _render_grid(
    key_index: int,
    palette: tuple[int, int, int, int],
    base_mask: np.ndarray,
    orientations: list[int],
) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    grid[DIVIDERS[0], :] = GREY
    grid[DIVIDERS[1], :] = GREY
    grid[:, DIVIDERS[0]] = GREY
    grid[:, DIVIDERS[1]] = GREY

    row_bands, col_bands = _cell_bands(grid)
    family = _canonical_rotation_family(base_mask)
    non_key_ptr = 0
    for idx, ((r0, r1), (c0, c1)) in enumerate(
        (bands for bands in ((rb, cb) for rb in row_bands for cb in col_bands))
    ):
        cell = grid[r0:r1, c0:c1]
        if idx == key_index:
            for color, (rr, cc) in zip(palette, CORNERS):
                cell[rr, cc] = int(color)
            continue
        mask = family[int(orientations[non_key_ptr])]
        cell[mask] = SPRITE_COLOR
        non_key_ptr += 1
    return grid


def _validate_input(grid: np.ndarray) -> None:
    cells, key_cell, family, orientations = _parse_task(grid)
    counts = [0, 0, 0, 0]
    for _cell_index, orientation in orientations:
        counts[orientation] += 1
    assert min(counts) >= 1, "every rotation must appear at least once"
    assert len({_mask_signature(member) for member in family}) == 4
    key_colors = [int(key_cell["cell"][r, c]) for r, c in CORNERS]
    assert len(set(key_colors)) == 4
    for cell_info in cells:
        if cell_info is key_cell:
            continue
        cell = cell_info["cell"]
        mask = (cell != BLACK) & (cell != GREY)
        assert int(mask.sum()) >= 5


def _expected_output(grid: np.ndarray) -> np.ndarray:
    return main(grid)


def _generate_orientations() -> list[int]:
    orientations = [0, 1, 2, 3]
    orientations.extend(random.randrange(4) for _ in range(4))
    random.shuffle(orientations)
    return orientations


def _gen_attempt() -> np.ndarray | None:
    key_index = random.randrange(9)
    palette = tuple(random.sample(PALETTE_COLORS, 4))
    base_mask = random.choice(BASE_MASKS)
    orientations = _generate_orientations()
    grid = _render_grid(key_index, palette, base_mask, orientations)
    try:
        _validate_input(grid)
        out = _expected_output(grid)
        cells, key_cell, _family, parsed_orientations = _parse_task(grid)
        key_palette = [int(key_cell["cell"][r, c]) for r, c in CORNERS]
        assert set(key_palette) == set(PALETTE_COLORS)
        for cell_info in cells:
            if cell_info is key_cell:
                continue
            r0, r1, c0, c1 = cell_info["slice"]
            before = grid[r0:r1, c0:c1]
            after = out[r0:r1, c0:c1]
            mask = (before != BLACK) & (before != GREY)
            assert np.all(after[mask] == key_palette[int(cell_info["orientation"])])
            assert np.array_equal(after[~mask], before[~mask])
        assert len(parsed_orientations) == 8
        return grid
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotation-palette ARC input")
