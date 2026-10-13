from __future__ import annotations

import random
import numpy as np


def _most_common_value(arr: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(arr, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _corners(n: int) -> set[tuple[int, int]]:
    return {(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)}


def _rotation_options(tile: np.ndarray) -> list[np.ndarray]:
    options: dict[bytes, np.ndarray] = {}
    for k in range(4):
        rot = np.rot90(tile, k)
        options[rot.tobytes()] = rot
    return list(options.values())


def _compatible(base: np.ndarray, patch: np.ndarray, bg: int) -> bool:
    overlap = (base != int(bg)) & (patch != int(bg))
    return not np.any(base[overlap] != patch[overlap])


def _merge(base: np.ndarray, patch: np.ndarray, bg: int) -> np.ndarray:
    out = base.copy()
    fill = (out == int(bg)) & (patch != int(bg))
    out[fill] = patch[fill]
    return out


def _find_separator_layout(grid: np.ndarray) -> tuple[int, list[int], list[int], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    candidates: list[tuple[int, list[int], list[int], int]] = []
    for color in map(int, np.unique(g)):
        row_hits = [r for r in range(h) if np.all(g[r, :] == color)]
        col_hits = [c for c in range(w) if np.all(g[:, c] == color)]
        if len(row_hits) < 3 or len(col_hits) < 3:
            continue
        if row_hits[0] != 0 or row_hits[-1] != h - 1:
            continue
        if col_hits[0] != 0 or col_hits[-1] != w - 1:
            continue
        row_steps = np.diff(row_hits)
        col_steps = np.diff(col_hits)
        if len(set(map(int, row_steps))) != 1 or len(set(map(int, col_steps))) != 1:
            continue
        step = int(row_steps[0])
        if step != int(col_steps[0]) or step < 2:
            continue
        if (h - 1) % step != 0 or (w - 1) % step != 0:
            continue
        candidates.append((color, row_hits, col_hits, step - 1))

    assert len(candidates) == 1, "expected one separator color with a square periodic grid"
    return candidates[0]


def _extract_tiles(grid: np.ndarray) -> tuple[list[np.ndarray], int, int]:
    g = np.asarray(grid, dtype=int)
    sep, row_hits, col_hits, tile_n = _find_separator_layout(g)

    tiles: list[np.ndarray] = []
    for br in range(len(row_hits) - 1):
        for bc in range(len(col_hits) - 1):
            r0 = row_hits[br] + 1
            r1 = row_hits[br + 1]
            c0 = col_hits[bc] + 1
            c1 = col_hits[bc + 1]
            tile = g[r0:r1, c0:c1].copy()
            assert tile.shape == (tile_n, tile_n)
            tiles.append(tile)

    non_sep = np.concatenate([tile[tile != int(sep)] for tile in tiles])
    bg = _most_common_value(non_sep)

    marker_idx = None
    corner_set = _corners(tile_n)
    for idx, tile in enumerate(tiles):
        coords = np.argwhere(tile == int(sep))
        if coords.size == 0:
            continue
        assert coords.shape == (1, 2), "expected one separator-colored marker pixel in exactly one tile"
        r, c = map(int, coords[0])
        assert (r, c) in corner_set, "marker pixel must occupy a tile corner"
        assert marker_idx is None, "expected a unique marker tile"
        marker_idx = idx
        tile[r, c] = int(bg)

    assert marker_idx is not None, "missing marker tile"
    return tiles, int(bg), int(marker_idx)


def _consistent_unions(tiles: list[np.ndarray], bg: int, marker_idx: int) -> list[np.ndarray]:
    order = [int(marker_idx)] + sorted(
        [i for i in range(len(tiles)) if i != int(marker_idx)],
        key=lambda i: -int(np.count_nonzero(tiles[i] != int(bg))),
    )
    states: dict[bytes, np.ndarray] = {tiles[int(marker_idx)].tobytes(): tiles[int(marker_idx)].copy()}

    for idx in order[1:]:
        options = _rotation_options(tiles[int(idx)])
        next_states: dict[bytes, np.ndarray] = {}
        for state in states.values():
            for option in options:
                if not _compatible(state, option, int(bg)):
                    continue
                merged = _merge(state, option, int(bg))
                next_states[merged.tobytes()] = merged
        states = next_states
        assert states, "no conflict-free rotational alignment exists"

    return list(states.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align all rotated fragments in the separator-grid mosaic to the marked tile's orientation and merge them into one complete tile.

    Input structure:
    - Separator-colored full rows and columns partition the canvas into equal square tiles, with a separator border around the outside.
    - Each tile contains a fragment of the same multicolor pattern, but each fragment may be rotated by 0/90/180/270 degrees and may miss some pixels to the tile background.
    - Exactly one tile contains one extra separator-colored pixel in a tile corner as the marker.

    Rule:
    - Use the marked tile's current orientation as the target orientation.
    - Remove the marker pixel, then rotate every other tile fragment into the marked tile's orientation.
    - Among all rotations, keep only alignments whose non-background pixels never disagree in the same cell, and merge all aligned fragments by foreground union.
    - Output the single recovered tile only; separator lines and the marker do not appear.

    Key insight: The marker chooses the final orientation, but the other tiles' rotations must be inferred from fragment-to-fragment consistency because they are all incomplete views of the same pattern.
    """
    tiles, bg, marker_idx = _extract_tiles(np.asarray(grid, dtype=int))
    unions = _consistent_unions(tiles, int(bg), int(marker_idx))
    assert len(unions) == 1, "fragment alignment must determine a unique merged pattern"
    return unions[0]


_BASE_SHAPES = [
    np.array([[1]], dtype=int),
    np.array([[1, 1]], dtype=int),
    np.array([[1], [1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
]


def _rand_palette() -> tuple[int, int, list[int]]:
    colors = list(range(10))
    bg, sep = random.sample(colors, 2)
    remaining = [c for c in colors if c not in {bg, sep}]
    fg_count = random.randint(2, 4)
    fg_colors = random.sample(remaining, fg_count)
    return int(bg), int(sep), [int(c) for c in fg_colors]


def _random_template(tile_n: int, bg: int, fg_colors: list[int], n_tiles: int) -> np.ndarray | None:
    corner_set = _corners(tile_n)
    min_fg = max(n_tiles + 4, tile_n + 3)
    max_fg = min(tile_n * tile_n - 4, 3 * tile_n)

    for _ in range(200):
        template = np.full((tile_n, tile_n), int(bg), dtype=int)
        feature_count = random.randint(4, 7)
        for _ in range(feature_count):
            mask = random.choice(_BASE_SHAPES)
            if random.choice([False, True]):
                mask = np.rot90(mask, random.randrange(4))
            mh, mw = mask.shape
            for _ in range(30):
                r0 = random.randint(0, tile_n - mh)
                c0 = random.randint(0, tile_n - mw)
                coords = [(r0 + r, c0 + c) for r in range(mh) for c in range(mw) if mask[r, c]]
                if any((r, c) in corner_set for r, c in coords):
                    continue
                if any(template[r, c] != int(bg) for r, c in coords):
                    continue
                color = random.choice(fg_colors)
                for r, c in coords:
                    template[r, c] = int(color)
                break

        fg_mask = template != int(bg)
        fg_count = int(np.count_nonzero(fg_mask))
        if fg_count < min_fg or fg_count > max_fg:
            continue
        if len(set(map(int, np.unique(template[fg_mask])))) < 2:
            continue
        if any(np.array_equal(template, np.rot90(template, k)) for k in (1, 2, 3)):
            continue
        return template

    return None


def _visibility_masks(template: np.ndarray, bg: int, n_tiles: int) -> list[np.ndarray] | None:
    fg_coords = [tuple(map(int, rc)) for rc in np.argwhere(template != int(bg))]
    if len(fg_coords) < n_tiles + 4:
        return None

    target_min = max(4, len(fg_coords) // 3)
    target_max = len(fg_coords) - 1

    for _ in range(200):
        random.shuffle(fg_coords)
        masks = [np.zeros_like(template, dtype=bool) for _ in range(n_tiles)]

        for idx in range(n_tiles):
            r, c = fg_coords[idx]
            masks[idx][r, c] = True

        for r, c in fg_coords[n_tiles:]:
            subset_size = random.randint(2, max(2, n_tiles - 1))
            for idx in random.sample(range(n_tiles), subset_size):
                masks[idx][r, c] = True

        counts = [int(np.count_nonzero(mask)) for mask in masks]
        if min(counts) < target_min:
            continue
        if max(counts) > target_max:
            continue
        if any(count == len(fg_coords) for count in counts):
            continue
        return masks

    return None


def _build_input(
    template: np.ndarray,
    bg: int,
    sep: int,
    tile_rows: int,
    tile_cols: int,
    rotations: list[int],
    marker_idx: int,
    marker_corner: tuple[int, int],
    masks: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    tile_n = int(template.shape[0])
    h = tile_rows * tile_n + tile_rows + 1
    w = tile_cols * tile_n + tile_cols + 1
    grid = np.full((h, w), int(sep), dtype=int)
    clean_tiles: list[np.ndarray] = []

    for idx in range(tile_rows * tile_cols):
        fragment = np.where(masks[idx], template, int(bg))
        tile = np.rot90(fragment, int(rotations[idx]))
        clean_tiles.append(tile.copy())
        if idx == int(marker_idx):
            mr, mc = marker_corner
            assert tile[mr, mc] == int(bg), "marker must sit on a true background corner"
            tile = tile.copy()
            tile[mr, mc] = int(sep)

        br = idx // tile_cols
        bc = idx % tile_cols
        r0 = br * (tile_n + 1) + 1
        c0 = bc * (tile_n + 1) + 1
        grid[r0 : r0 + tile_n, c0 : c0 + tile_n] = tile

    return grid, clean_tiles


def _validate_instance(
    grid: np.ndarray,
    bg: int,
    marker_idx: int,
    expected: np.ndarray,
) -> bool:
    try:
        parsed_tiles, parsed_bg, parsed_marker = _extract_tiles(grid)
        if parsed_bg != int(bg) or parsed_marker != int(marker_idx):
            return False

        unions = _consistent_unions(parsed_tiles, int(parsed_bg), int(parsed_marker))
        if len(unions) != 1 or not np.array_equal(unions[0], expected):
            return False
        return np.array_equal(main(grid), expected)
    except AssertionError:
        return False


def _semantic_checks(
    template: np.ndarray,
    masks: list[np.ndarray],
    rotations: list[int],
    marker_idx: int,
    expected: np.ndarray,
    bg: int,
) -> bool:
    if len(set(map(int, rotations))) < 2:
        return False
    if any(np.array_equal(template, np.rot90(template, k)) for k in (1, 2, 3)):
        return False

    ref_rot = int(rotations[marker_idx])
    aligned_tiles = []
    for mask in masks:
        fragment = np.where(mask, template, int(bg))
        aligned_tiles.append(np.rot90(fragment, ref_rot))

    union = np.full_like(expected, int(bg))
    for tile in aligned_tiles:
        if not _compatible(union, tile, int(bg)):
            return False
        union = _merge(union, tile, int(bg))
    if not np.array_equal(union, expected):
        return False

    for idx in range(len(aligned_tiles)):
        partial = np.full_like(expected, int(bg))
        for jdx, tile in enumerate(aligned_tiles):
            if idx == jdx:
                continue
            partial = _merge(partial, tile, int(bg))
        if np.array_equal(partial, expected):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    tile_rows, tile_cols = random.choice([(2, 2), (2, 3), (3, 2)])
    n_tiles = tile_rows * tile_cols
    max_tile_n = min((30 - (tile_rows + 1)) // tile_rows, (30 - (tile_cols + 1)) // tile_cols)
    tile_n = random.randint(5, max(5, min(8, max_tile_n)))

    bg, sep, fg_colors = _rand_palette()
    template = _random_template(tile_n, int(bg), fg_colors, n_tiles)
    if template is None:
        return None

    masks = _visibility_masks(template, int(bg), n_tiles)
    if masks is None:
        return None

    rotations = [random.randrange(4) for _ in range(n_tiles)]
    if len(set(rotations)) < 2:
        rotations[random.randrange(n_tiles)] = (rotations[0] + random.randint(1, 3)) % 4

    marker_idx = random.randrange(n_tiles)
    marker_corner = random.choice(sorted(_corners(tile_n)))
    expected = np.rot90(template, int(rotations[marker_idx]))

    if not _semantic_checks(template, masks, rotations, marker_idx, expected, int(bg)):
        return None

    grid, _ = _build_input(
        template,
        int(bg),
        int(sep),
        tile_rows,
        tile_cols,
        rotations,
        int(marker_idx),
        marker_corner,
        masks,
    )

    if not _validate_instance(grid, int(bg), int(marker_idx), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid residual-fragment alignment puzzle")
