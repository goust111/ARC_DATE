import random
from typing import Iterable

import numpy as np


PALETTE = tuple(range(10))
TRANSFORM_COUNT = 8
MAX_GENERATION_ATTEMPTS = 400
TILE_SIZE_RANGE = (5, 8)
GRID_LAYOUT_OPTIONS = tuple(
    (tile_size, tile_rows, tile_cols)
    for tile_size in range(TILE_SIZE_RANGE[0], TILE_SIZE_RANGE[1] + 1)
    for tile_rows in range(2, 5)
    for tile_cols in range(2, 5)
    if tile_rows * tile_cols >= 4
    and tile_rows * tile_size + tile_rows + 1 <= 30
    and tile_cols * tile_size + tile_cols + 1 <= 30
)


def _apply_transform(tile: np.ndarray, transform_idx: int) -> np.ndarray:
    out = np.rot90(tile, transform_idx % 4)
    if transform_idx >= 4:
        out = np.fliplr(out)
    return out


def _transform_coord(coord: tuple[int, int], size: int, transform_idx: int) -> tuple[int, int]:
    r, c = map(int, coord)
    for _ in range(transform_idx % 4):
        r, c = size - 1 - c, r
    if transform_idx >= 4:
        c = size - 1 - c
    return int(r), int(c)


def _segments(mask: np.ndarray) -> list[tuple[bool, int, int]]:
    assert mask.ndim == 1 and mask.size > 0
    segments = []
    start = 0
    current = bool(mask[0])
    for idx in range(1, int(mask.size)):
        value = bool(mask[idx])
        if value != current:
            segments.append((current, start, idx - 1))
            start = idx
            current = value
    segments.append((current, start, int(mask.size) - 1))
    return segments


def _parse_separator_layout(grid: np.ndarray) -> dict:
    h, w = map(int, grid.shape)
    candidates = []
    for color in map(int, np.unique(grid)):
        row_mask = np.all(grid == color, axis=1)
        col_mask = np.all(grid == color, axis=0)
        if not row_mask.any() or not col_mask.any():
            continue

        row_segments = _segments(row_mask)
        col_segments = _segments(col_mask)
        row_tiles = [(r0, r1) for is_sep, r0, r1 in row_segments if not is_sep]
        col_tiles = [(c0, c1) for is_sep, c0, c1 in col_segments if not is_sep]
        row_seps = [(r0, r1) for is_sep, r0, r1 in row_segments if is_sep]
        col_seps = [(c0, c1) for is_sep, c0, c1 in col_segments if is_sep]
        if not row_tiles or not col_tiles or not row_seps or not col_seps:
            continue

        row_sizes = {r1 - r0 + 1 for r0, r1 in row_tiles}
        col_sizes = {c1 - c0 + 1 for c0, c1 in col_tiles}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue
        tile_size = row_sizes.pop()
        if tile_size != col_sizes.pop() or tile_size < 2:
            continue

        row_sep_sizes = {r1 - r0 + 1 for r0, r1 in row_seps}
        col_sep_sizes = {c1 - c0 + 1 for c0, c1 in col_seps}
        if len(row_sep_sizes) != 1 or len(col_sep_sizes) != 1:
            continue
        if row_sep_sizes != col_sep_sizes:
            continue

        covered_rows = sum(r1 - r0 + 1 for r0, r1 in row_tiles) + sum(
            r1 - r0 + 1 for r0, r1 in row_seps
        )
        covered_cols = sum(c1 - c0 + 1 for c0, c1 in col_tiles) + sum(
            c1 - c0 + 1 for c0, c1 in col_seps
        )
        if covered_rows != h or covered_cols != w:
            continue

        tiles = [
            [grid[r0 : r1 + 1, c0 : c1 + 1].copy() for c0, c1 in col_tiles]
            for r0, r1 in row_tiles
        ]
        candidates.append(
            {
                "separator_color": color,
                "tile_size": tile_size,
                "row_tiles": row_tiles,
                "col_tiles": col_tiles,
                "tiles": tiles,
            }
        )

    if not candidates:
        raise ValueError("could not detect separator layout")
    candidates.sort(
        key=lambda item: (
            len(item["row_tiles"]) * len(item["col_tiles"]),
            item["tile_size"],
        ),
        reverse=True,
    )
    return candidates[0]


def _majority_template(normalized_tiles: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(normalized_tiles, axis=0)
    tile_count, tile_size, _ = stack.shape
    template = np.zeros((tile_size, tile_size), dtype=int)
    for r in range(tile_size):
        for c in range(tile_size):
            counts = np.bincount(stack[:, r, c], minlength=10)
            color = int(np.argmax(counts))
            if counts[color] * 2 <= tile_count:
                raise ValueError("no strict majority at a tile coordinate")
            template[r, c] = color
    return template


def _find_transform_to_target(
    coord: tuple[int, int], target: tuple[int, int], size: int
) -> int | None:
    matches = [
        transform_idx
        for transform_idx in range(TRANSFORM_COUNT)
        if _transform_coord(coord, size, transform_idx) == target
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _flatten_tiles(tiles: list[list[np.ndarray]]) -> list[np.ndarray]:
    return [tile for row in tiles for tile in row]


def _infer_repair(flat_tiles: list[np.ndarray]) -> dict:
    tile_size = int(flat_tiles[0].shape[0])
    colors = sorted({int(color) for tile in flat_tiles for color in np.unique(tile)})
    valid_solutions = []

    for anchor_color in colors:
        anchor_coords = []
        for tile in flat_tiles:
            coords = np.argwhere(tile == anchor_color)
            if coords.shape[0] != 1:
                anchor_coords = []
                break
            anchor_coords.append((int(coords[0, 0]), int(coords[0, 1])))
        if not anchor_coords:
            continue

        orbit = {
            _transform_coord(anchor_coords[0], tile_size, transform_idx)
            for transform_idx in range(TRANSFORM_COUNT)
        }
        if len(orbit) != TRANSFORM_COUNT:
            continue
        canonical_anchor = min(orbit)

        normalized_tiles = []
        normalize_transforms = []
        valid = True
        for tile, coord in zip(flat_tiles, anchor_coords):
            transform_idx = _find_transform_to_target(coord, canonical_anchor, tile_size)
            if transform_idx is None:
                valid = False
                break
            normalized_tiles.append(_apply_transform(tile, transform_idx))
            normalize_transforms.append(transform_idx)
        if not valid:
            continue

        try:
            template = _majority_template(normalized_tiles)
        except ValueError:
            continue
        if int(template[canonical_anchor]) != anchor_color:
            continue

        mismatch_masks = [tile != template for tile in normalized_tiles]
        damaged_indices = [idx for idx, mask in enumerate(mismatch_masks) if bool(mask.any())]
        if len(damaged_indices) != 1:
            continue

        damaged_index = damaged_indices[0]
        diff_mask = mismatch_masks[damaged_index]
        replacement_colors = np.unique(normalized_tiles[damaged_index][diff_mask])
        if replacement_colors.size != 1:
            continue
        replacement_color = int(replacement_colors[0])
        if replacement_color == anchor_color:
            continue
        if np.any(template[diff_mask] == replacement_color):
            continue

        valid_solutions.append(
            {
                "anchor_color": anchor_color,
                "canonical_anchor": canonical_anchor,
                "template": template,
                "damaged_index": damaged_index,
                "anchor_coords": anchor_coords,
                "normalize_transforms": normalize_transforms,
            }
        )

    if not valid_solutions:
        raise ValueError("could not infer anchor color and damaged tile")
    return valid_solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the unique damaged tile by canonicalizing all anchor-marked copies and voting for the shared template.

    Input structure:
    - Full separator-color rows and columns partition the grid into equal square tiles.
    - Every tile is a rotation or mirror of one multicolor template that contains exactly one anchor-color pixel.
    - Exactly one tile is damaged: some non-anchor template pixels were replaced by the tile background color.

    Rule:
    - Detect the separator layout and extract the square tiles.
    - Use the anchor position in each tile to transform all copies into one shared canonical orientation.
    - Take a per-cell majority vote to recover the undamaged template, then transform that template back into the damaged tile's own orientation and paste it into the original grid.

    Key insight:
    - Majority voting only becomes meaningful after orientation normalization, and the anchor pixel makes that normalization unique.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    layout = _parse_separator_layout(g)
    flat_tiles = _flatten_tiles(layout["tiles"])
    repair = _infer_repair(flat_tiles)

    damaged_index = int(repair["damaged_index"])
    row_count = len(layout["row_tiles"])
    col_count = len(layout["col_tiles"])
    damage_row = damaged_index // col_count
    damage_col = damaged_index % col_count
    tile_r0, tile_r1 = layout["row_tiles"][damage_row]
    tile_c0, tile_c1 = layout["col_tiles"][damage_col]

    observed_anchor = repair["anchor_coords"][damaged_index]
    canonical_anchor = repair["canonical_anchor"]
    restore_transform = _find_transform_to_target(
        canonical_anchor,
        observed_anchor,
        layout["tile_size"],
    )
    if restore_transform is None:
        raise ValueError("could not reorient repaired template")
    repaired_tile = _apply_transform(repair["template"], restore_transform)

    out = g.copy()
    out[tile_r0 : tile_r1 + 1, tile_c0 : tile_c1 + 1] = repaired_tile
    return out


def _neighbors4(cell: tuple[int, int]) -> Iterable[tuple[int, int]]:
    r, c = cell
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _normalize_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _random_connected_shape(size: int) -> list[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = {
            neighbor
            for cell in cells
            for neighbor in _neighbors4(cell)
            if neighbor not in cells
        }
        cells.add(random.choice(tuple(frontier)))
    return _normalize_cells(cells)


def _anchor_positions(tile_size: int) -> list[tuple[int, int]]:
    positions = []
    for r in range(tile_size):
        for c in range(tile_size):
            orbit = {
                _transform_coord((r, c), tile_size, transform_idx)
                for transform_idx in range(TRANSFORM_COUNT)
            }
            if len(orbit) == TRANSFORM_COUNT:
                positions.append((r, c))
    return positions


def _place_shape(
    tile: np.ndarray,
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    color: int,
    size: int,
) -> bool:
    tile_size = int(tile.shape[0])
    for _ in range(80):
        shape = _random_connected_shape(size)
        shape_h = max(r for r, _ in shape) + 1
        shape_w = max(c for _, c in shape) + 1
        if shape_h > tile_size or shape_w > tile_size:
            continue
        r0 = random.randint(0, tile_size - shape_h)
        c0 = random.randint(0, tile_size - shape_w)
        cells = {(r0 + r, c0 + c) for r, c in shape}
        if cells & occupied or cells & blocked:
            continue
        for r, c in cells:
            tile[r, c] = color
        occupied.update(cells)
        return True
    return False


def _build_template(tile_size: int, tile_bg: int, anchor_color: int, motif_colors: list[int]) -> np.ndarray | None:
    tile = np.full((tile_size, tile_size), tile_bg, dtype=int)
    anchor = random.choice(_anchor_positions(tile_size))
    tile[anchor] = anchor_color
    occupied = {anchor}
    blocked = {anchor}

    for color in motif_colors:
        if not _place_shape(tile, occupied, blocked, color, random.randint(2, 4)):
            return None

    extra_shapes = random.randint(1, 3)
    for _ in range(extra_shapes):
        color = random.choice(motif_colors)
        if not _place_shape(tile, occupied, blocked, color, random.randint(1, 3)):
            return None

    counts = {color: int(np.count_nonzero(tile == color)) for color in motif_colors}
    if any(count < 2 for count in counts.values()):
        return None
    if int(np.count_nonzero((tile != tile_bg) & (tile != anchor_color))) < 6:
        return None
    return tile


def _render_grid(
    tiles: list[np.ndarray],
    tile_rows: int,
    tile_cols: int,
    tile_size: int,
    separator_color: int,
) -> np.ndarray:
    height = tile_rows * tile_size + tile_rows + 1
    width = tile_cols * tile_size + tile_cols + 1
    grid = np.full((height, width), separator_color, dtype=int)
    idx = 0
    for tile_r in range(tile_rows):
        for tile_c in range(tile_cols):
            r0 = 1 + tile_r * (tile_size + 1)
            c0 = 1 + tile_c * (tile_size + 1)
            grid[r0 : r0 + tile_size, c0 : c0 + tile_size] = tiles[idx]
            idx += 1
    return grid


def _damage_tile(tile: np.ndarray, tile_bg: int, anchor_color: int) -> np.ndarray | None:
    damageable = [
        (int(r), int(c))
        for r, c in np.argwhere((tile != tile_bg) & (tile != anchor_color))
    ]
    if len(damageable) < 2:
        return None
    damaged = tile.copy()
    damage_count = random.randint(1, min(4, len(damageable) - 1))
    for r, c in random.sample(damageable, damage_count):
        damaged[r, c] = tile_bg
    return damaged


def _generate_instance() -> tuple[np.ndarray, np.ndarray] | None:
    tile_size, tile_rows, tile_cols = random.choice(GRID_LAYOUT_OPTIONS)
    tile_count = tile_rows * tile_cols

    colors = random.sample(PALETTE, random.randint(5, 6))
    separator_color, tile_bg, anchor_color = colors[:3]
    motif_colors = colors[3:]

    template = None
    for _ in range(120):
        template = _build_template(tile_size, tile_bg, anchor_color, motif_colors)
        if template is not None:
            break
    if template is None:
        return None

    orientations = [random.randrange(TRANSFORM_COUNT) for _ in range(tile_count)]
    if len(set(orientations)) < 2:
        return None

    pristine_tiles = [_apply_transform(template, orientation) for orientation in orientations]
    damaged_index = random.randrange(tile_count)
    damaged_tile = _damage_tile(pristine_tiles[damaged_index], tile_bg, anchor_color)
    if damaged_tile is None:
        return None

    input_tiles = [tile.copy() for tile in pristine_tiles]
    input_tiles[damaged_index] = damaged_tile

    inp = _render_grid(input_tiles, tile_rows, tile_cols, tile_size, separator_color)
    out = _render_grid(pristine_tiles, tile_rows, tile_cols, tile_size, separator_color)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        generated = _generate_instance()
        if generated is None:
            continue
        inp, out = generated
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, out):
            return inp
    raise RuntimeError("failed to generate a valid input")
