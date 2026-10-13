from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _non_separator_runs(full_separator: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_sep in enumerate(full_separator.tolist()):
        if is_sep:
            if start is not None:
                runs.append((start, idx - 1))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        runs.append((start, len(full_separator) - 1))
    return runs


def _parse_lattice(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], list[list[np.ndarray]]]:
    sep = int(grid[0, 0])
    row_runs = _non_separator_runs(np.all(grid == sep, axis=1))
    col_runs = _non_separator_runs(np.all(grid == sep, axis=0))
    assert row_runs and col_runs, "expected separator lines that define tiles"

    row_heights = {r1 - r0 + 1 for r0, r1 in row_runs}
    col_widths = {c1 - c0 + 1 for c0, c1 in col_runs}
    assert len(row_heights) == 1 and len(col_widths) == 1, "all tiles must share one size"
    assert next(iter(row_heights)) == next(iter(col_widths)), "tiles must be square"

    tiles: list[list[np.ndarray]] = []
    for r0, r1 in row_runs:
        row_tiles = []
        for c0, c1 in col_runs:
            row_tiles.append(grid[r0 : r1 + 1, c0 : c1 + 1].copy())
        tiles.append(row_tiles)
    return sep, row_runs, col_runs, tiles


def _eye_position(tile: np.ndarray, eye_color: int) -> tuple[int, int]:
    coords = np.argwhere(tile == eye_color)
    assert coords.shape[0] == 1, "every tile must contain exactly one eye"
    return int(coords[0, 0]), int(coords[0, 1])


def _dihedral_variants(tile: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    ordered = [np.rot90(tile, k) for k in range(4)]
    ordered.extend(np.fliplr(np.rot90(tile, k)) for k in range(4))
    for transformed in ordered:
        key = transformed.tobytes()
        if key not in seen:
            seen.add(key)
            variants.append(transformed.copy())
    return variants


def _infer_eye_color(tiles: list[list[np.ndarray]], sep: int) -> int:
    flat_tiles = [tile for row in tiles for tile in row]
    colors = set(map(int, np.unique(np.stack(flat_tiles))))
    colors.discard(sep)

    candidates = []
    for color in sorted(colors):
        if all(int(np.count_nonzero(tile == color)) == 1 for tile in flat_tiles):
            candidates.append(color)
    assert len(candidates) == 1, "expected one unique per-tile eye color"
    return int(candidates[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the fully occluded tile from the surviving eye pixel and the visible rigid transforms.

    Input structure:
    - Separator lines divide the canvas into equal square tiles.
    - Every visible tile contains the same asymmetric motif on the same background, using one body color plus one unique eye color.
    - One tile is covered by a solid occlusion color everywhere except for its eye pixel, which stays in its original tile-relative position.

    Rule:
    - Parse the lattice and identify the eye color as the only non-separator color that appears exactly once in every tile.
    - Take any visible tile as a reference motif, enumerate its eight rotation/reflection variants, and verify that every other visible tile is one of those rigid transforms.
    - In the occluded tile, match the surviving eye position to the unique variant whose eye lands at that same tile-relative coordinate, then replace the whole occluded tile with that variant.

    Key insight:
    - The eye is rigidly attached to the motif, so its tile-relative position encodes the tile orientation even when the rest of that tile has been hidden.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep, row_runs, col_runs, tiles = _parse_lattice(g)
    eye_color = _infer_eye_color(tiles, sep)

    visible_positions: list[tuple[int, int]] = []
    hidden_positions: list[tuple[int, int]] = []
    for tr, row in enumerate(tiles):
        for tc, tile in enumerate(row):
            remaining = set(map(int, np.unique(tile))) - {sep, eye_color}
            if len(remaining) == 1:
                hidden_positions.append((tr, tc))
            else:
                visible_positions.append((tr, tc))

    assert len(hidden_positions) == 1, "expected exactly one occluded tile"
    assert visible_positions, "need at least one visible tile"

    ref_r, ref_c = visible_positions[0]
    ref_tile = tiles[ref_r][ref_c]
    variants = _dihedral_variants(ref_tile)
    assert len(variants) >= 2, "reference tile must admit orientation changes"

    by_bytes = {variant.tobytes(): variant for variant in variants}
    for tr, tc in visible_positions:
        assert tiles[tr][tc].tobytes() in by_bytes, "visible tile is not a rigid transform of the motif"

    hidden_r, hidden_c = hidden_positions[0]
    hidden_tile = tiles[hidden_r][hidden_c]
    hidden_eye = _eye_position(hidden_tile, eye_color)

    matches = [variant for variant in variants if _eye_position(variant, eye_color) == hidden_eye]
    assert len(matches) == 1, "eye position must determine a unique orientation"

    out = g.copy()
    r0, r1 = row_runs[hidden_r]
    c0, c1 = col_runs[hidden_c]
    out[r0 : r1 + 1, c0 : c1 + 1] = matches[0]
    return out


_GEN_TILE_ROWS = (3, 4)
_GEN_TILE_COLS = (3, 4)
_GEN_TILE_SIZE = (4, 6)
_GEN_MAX_ATTEMPTS = 400
_GEN_PATTERN_CELLS = (5, 10)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _grow_connected_mask(size: int, count: int) -> np.ndarray | None:
    mask = np.zeros((size, size), dtype=bool)
    start = (random.randrange(size), random.randrange(size))
    mask[start] = True

    while int(mask.sum()) < count:
        frontier = []
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < size and 0 <= nc < size and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        mask[random.choice(frontier)] = True
    return mask


def _base_tile_variants(tile: np.ndarray, eye_color: int) -> list[np.ndarray] | None:
    variants = _dihedral_variants(tile)
    if len(variants) != 8:
        return None
    eyes = [_eye_position(variant, eye_color) for variant in variants]
    if len(set(eyes)) != 8:
        return None
    return variants


def _make_base_tile(tile_size: int, bg: int, body: int, eye: int) -> tuple[np.ndarray, list[np.ndarray]] | None:
    low = max(_GEN_PATTERN_CELLS[0], tile_size + 1)
    high = min(_GEN_PATTERN_CELLS[1], tile_size * tile_size - 2)
    if low > high:
        return None

    for _ in range(120):
        count = random.randint(low, high)
        mask = _grow_connected_mask(tile_size, count)
        if mask is None:
            continue
        box = _bbox(mask)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        if r1 - r0 + 1 < min(3, tile_size) or c1 - c0 + 1 < min(3, tile_size):
            continue
        if int(mask.sum()) >= tile_size * tile_size - 1:
            continue

        occupied = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
        eye_pos = random.choice(occupied)
        tile = np.full((tile_size, tile_size), bg, dtype=int)
        tile[mask] = body
        tile[eye_pos] = eye

        variants = _base_tile_variants(tile, eye)
        if variants is None:
            continue
        return tile, variants
    return None


def _sample_orientation_set(total_visible: int) -> list[int] | None:
    max_count = min(8, total_visible)
    if max_count < 4:
        return None
    count = random.randint(4, max_count)

    rotations = [0, 1, 2, 3]
    reflections = [4, 5, 6, 7]
    chosen = [random.choice(rotations), random.choice(reflections)]
    pool = [idx for idx in range(8) if idx not in chosen]
    random.shuffle(pool)
    chosen.extend(pool[: count - 2])
    random.shuffle(chosen)
    return chosen


def _render_lattice(tile_grid: list[list[np.ndarray]], sep: int) -> np.ndarray:
    rows = len(tile_grid)
    cols = len(tile_grid[0])
    tile_size = tile_grid[0][0].shape[0]
    stride = tile_size + 1
    canvas = np.full((rows * stride + 1, cols * stride + 1), sep, dtype=int)
    for tr in range(rows):
        for tc in range(cols):
            r0 = 1 + tr * stride
            c0 = 1 + tc * stride
            canvas[r0 : r0 + tile_size, c0 : c0 + tile_size] = tile_grid[tr][tc]
    return canvas


def _sample_scene() -> tuple[np.ndarray, np.ndarray] | None:
    tile_rows = random.randint(*_GEN_TILE_ROWS)
    tile_cols = random.randint(*_GEN_TILE_COLS)
    tile_size = random.randint(*_GEN_TILE_SIZE)
    if max(tile_rows * (tile_size + 1) + 1, tile_cols * (tile_size + 1) + 1) > 30:
        return None

    colors = random.sample(range(1, 10), 4)
    bg, body, eye, occlusion = colors
    built = _make_base_tile(tile_size, bg, body, eye)
    if built is None:
        return None
    _base_tile, variants = built

    total_tiles = tile_rows * tile_cols
    visible_count = total_tiles - 1
    visible_orients = _sample_orientation_set(visible_count)
    if visible_orients is None:
        return None

    unseen = [idx for idx in range(8) if idx not in visible_orients]
    if unseen and random.random() < 0.5:
        hidden_orient = random.choice(unseen)
    else:
        hidden_orient = random.choice(visible_orients)

    positions = [(r, c) for r in range(tile_rows) for c in range(tile_cols)]
    hidden_pos = random.choice(positions)
    visible_positions = [pos for pos in positions if pos != hidden_pos]
    random.shuffle(visible_positions)

    orient_map: dict[tuple[int, int], int] = {hidden_pos: hidden_orient}
    for pos, orient in zip(visible_positions, visible_orients):
        orient_map[pos] = orient
    for pos in visible_positions[len(visible_orients) :]:
        orient_map[pos] = random.choice(visible_orients)

    tile_grid_out: list[list[np.ndarray]] = []
    tile_grid_in: list[list[np.ndarray]] = []
    for tr in range(tile_rows):
        out_row = []
        in_row = []
        for tc in range(tile_cols):
            tile = variants[orient_map[(tr, tc)]].copy()
            out_row.append(tile)
            if (tr, tc) == hidden_pos:
                occluded = np.full(tile.shape, occlusion, dtype=int)
                er, ec = _eye_position(tile, eye)
                occluded[er, ec] = eye
                in_row.append(occluded)
            else:
                in_row.append(tile.copy())
        tile_grid_out.append(out_row)
        tile_grid_in.append(in_row)

    inp = _render_lattice(tile_grid_in, sep=0)
    out = _render_lattice(tile_grid_out, sep=0)

    if np.array_equal(inp, out):
        return None
    return inp, out


def _validate_generated(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        sep, row_runs, col_runs, tiles = _parse_lattice(inp)
        eye_color = _infer_eye_color(tiles, sep)
    except Exception:
        return False

    if len(row_runs) < 3 or len(col_runs) < 3:
        return False

    flat_tiles = [tile for row in tiles for tile in row]
    hidden_indices = []
    visible_tiles = []
    for idx, tile in enumerate(flat_tiles):
        remaining = set(map(int, np.unique(tile))) - {sep, eye_color}
        if len(remaining) == 1:
            hidden_indices.append(idx)
        elif len(remaining) == 2:
            visible_tiles.append(tile)
        else:
            return False
    if len(hidden_indices) != 1 or len(visible_tiles) < 4:
        return False

    ref_variants = _dihedral_variants(visible_tiles[0])
    if len(ref_variants) != 8:
        return False
    eye_positions = [_eye_position(tile, eye_color) for tile in ref_variants]
    if len(set(eye_positions)) != 8:
        return False

    variant_set = {tile.tobytes() for tile in ref_variants}
    visible_unique = set()
    parities = set()
    for tile in visible_tiles:
        key = tile.tobytes()
        if key not in variant_set:
            return False
        visible_unique.add(key)
        for idx, variant in enumerate(ref_variants):
            if np.array_equal(tile, variant):
                parities.add(idx >= 4)
                break
    if len(visible_unique) < 4 or parities != {False, True}:
        return False

    hidden_tile = flat_tiles[hidden_indices[0]]
    hidden_eye = _eye_position(hidden_tile, eye_color)
    if sum(pos == hidden_eye for pos in eye_positions) != 1:
        return False

    solved = main(inp)
    if not np.array_equal(solved, out):
        return False
    if np.array_equal(inp, solved):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        sampled = _sample_scene()
        if sampled is None:
            continue
        inp, out = sampled
        if _validate_generated(inp, out):
            return inp
    raise RuntimeError("failed to generate a valid survivor-eye puzzle")
