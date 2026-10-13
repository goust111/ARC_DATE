from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _line_groups(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    groups: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for idx in map(int, indices[1:]):
        if idx == prev + 1:
            prev = idx
        else:
            groups.append((start, prev))
            start = prev = idx
    groups.append((start, prev))
    return groups


def _segments(length: int, separator_indices: list[int]) -> list[tuple[int, int]]:
    groups = _line_groups(sorted(separator_indices))
    out: list[tuple[int, int]] = []
    cursor = 0
    for start, stop in groups:
        if cursor <= start - 1:
            out.append((cursor, start - 1))
        cursor = stop + 1
    if cursor <= length - 1:
        out.append((cursor, length - 1))
    return out


def _same_span_size(spans: list[tuple[int, int]]) -> bool:
    if not spans:
        return False
    sizes = {stop - start + 1 for start, stop in spans}
    return len(sizes) == 1 and next(iter(sizes)) > 0


def _extract_tiles(
    grid: np.ndarray,
) -> tuple[list[np.ndarray], tuple[int, int], tuple[int, int], int]:
    height, width = map(int, grid.shape)
    candidates = []

    for color in map(int, np.unique(grid)):
        full_rows = [r for r in range(height) if np.all(grid[r, :] == color)]
        full_cols = [c for c in range(width) if np.all(grid[:, c] == color)]
        if not full_rows or not full_cols:
            continue

        row_spans = _segments(height, full_rows)
        col_spans = _segments(width, full_cols)
        if not (_same_span_size(row_spans) and _same_span_size(col_spans)):
            continue

        tile_count = len(row_spans) * len(col_spans)
        if tile_count < 3 or tile_count % 2 != 1:
            continue

        tiles = [
            grid[r0 : r1 + 1, c0 : c1 + 1].copy()
            for r0, r1 in row_spans
            for c0, c1 in col_spans
        ]
        tile_values = set(map(int, np.concatenate([tile.ravel() for tile in tiles])))
        if color in tile_values or len(tile_values) != 2:
            continue
        if any(len(set(map(int, tile.ravel()))) != 2 for tile in tiles):
            continue

        tile_shape = tiles[0].shape
        if any(tile.shape != tile_shape for tile in tiles):
            continue
        candidates.append((tiles, row_spans[0], col_spans[0], color))

    assert len(candidates) == 1, "expected one full-row/full-column separator layout"
    return candidates[0]


def _tile_colors(tiles: list[np.ndarray]) -> tuple[int, int]:
    colors = sorted(set(map(int, np.concatenate([tile.ravel() for tile in tiles]))))
    assert len(colors) == 2, "expected exactly two tile colors"
    return int(colors[0]), int(colors[1])


def _swap_colors(tile: np.ndarray, colors: tuple[int, int]) -> np.ndarray:
    a, b = colors
    assert np.all((tile == a) | (tile == b)), "tile contains a non-pattern color"
    return np.where(tile == a, b, a)


def _partner(tile: np.ndarray, colors: tuple[int, int]) -> np.ndarray:
    return np.rot90(_swap_colors(tile, colors), 2).copy()


def _tile_key(tile: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, tile.shape)), np.ascontiguousarray(tile).tobytes()


def _find_unpaired_tile(tiles: list[np.ndarray], colors: tuple[int, int]) -> np.ndarray:
    counts: Counter[tuple[tuple[int, int], bytes]] = Counter()
    examples: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for tile in tiles:
        key = _tile_key(tile)
        counts[key] += 1
        examples.setdefault(key, tile)

    seen: set[tuple[tuple[int, int], bytes]] = set()
    unpaired: list[np.ndarray] = []
    for key, count in counts.items():
        if key in seen:
            continue
        tile = examples[key]
        partner_key = _tile_key(_partner(tile, colors))
        partner_count = counts.get(partner_key, 0)

        if partner_key == key:
            if count % 2 == 1:
                unpaired.append(tile)
            seen.add(key)
            continue

        if count > partner_count:
            assert count - partner_count == 1, "expected only one unmatched tile"
            unpaired.append(tile)
        elif partner_count > count:
            assert partner_count - count == 1, "expected only one unmatched tile"
            unpaired.append(examples[partner_key])
        seen.add(key)
        seen.add(partner_key)

    assert len(unpaired) == 1, "expected exactly one tile without its composite partner"
    return unpaired[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the missing composite partner of the single unpaired tile.

    Input structure:
    - Full separator rows and columns split the grid into an odd number of equal-size tiles.
    - Every tile uses the same two pattern colors, while the separators use a third color.
    - All tiles except one can be paired with another tile equal to its color complement
      rotated 180 degrees.

    Rule:
    - Detect the separator rows and columns, then crop every equal-size tile.
    - Pair tiles by swapping the two tile colors and rotating the result 180 degrees.
    - Return the complement-then-rotated partner of the unique tile left without a match.

    Key insight:
    - The missing tile is not found by identity, plain complement, or plain rotation; the
      necessary relation is the composite of occupancy inversion and 180-degree rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    tiles, _row0, _col0, _sep_color = _extract_tiles(g)
    colors = _tile_colors(tiles)
    unpaired = _find_unpaired_tile(tiles, colors)
    return _partner(unpaired, colors)


# ---------------- generator ----------------

_COLORS = list(range(10))
_LAYOUTS = [(3, 3), (3, 5), (5, 3), (5, 5)]
_TILE_HEIGHTS = [3, 4, 5]
_TILE_WIDTHS = [3, 4, 5]
_MAX_ATTEMPTS = 500


def _mask_key(mask: np.ndarray) -> bytes:
    return np.ascontiguousarray(mask.astype(np.uint8)).tobytes()


def _mask_partner(mask: np.ndarray) -> np.ndarray:
    return np.rot90(~mask, 2).copy()


def _mask_valid(mask: np.ndarray) -> bool:
    area = int(mask.size)
    filled = int(np.count_nonzero(mask))
    if filled < 2 or filled > area - 2:
        return False
    if np.array_equal(mask, np.rot90(mask, 2)):
        return False
    if np.array_equal(mask, _mask_partner(mask)):
        return False
    return True


def _random_mask(height: int, width: int, used: set[bytes]) -> np.ndarray | None:
    for _ in range(300):
        density = random.uniform(0.30, 0.70)
        mask = np.random.random((height, width)) < density
        partner = _mask_partner(mask)
        if not (_mask_valid(mask) and _mask_valid(partner)):
            continue
        if _mask_key(mask) in used or _mask_key(partner) in used:
            continue
        return mask
    return None


def _make_counterexample_unpaired(
    used_masks: list[np.ndarray], used_keys: set[bytes]
) -> np.ndarray | None:
    options = used_masks[:]
    random.shuffle(options)
    for base in options:
        variants = [~base, np.rot90(base, 2)]
        random.shuffle(variants)
        for mask in variants:
            mask = mask.copy()
            partner = _mask_partner(mask)
            if not _mask_valid(mask):
                continue
            if _mask_key(mask) in used_keys or _mask_key(partner) in used_keys:
                continue
            return mask
    return None


def _paint_mask(mask: np.ndarray, colors: tuple[int, int]) -> np.ndarray:
    background, foreground = colors
    return np.where(mask, foreground, background).astype(int)


def _render_grid(
    masks: list[np.ndarray],
    layout: tuple[int, int],
    tile_shape: tuple[int, int],
    tile_colors: tuple[int, int],
    separator_color: int,
) -> np.ndarray:
    tile_rows, tile_cols = layout
    tile_h, tile_w = tile_shape
    height = tile_rows * tile_h + tile_rows - 1
    width = tile_cols * tile_w + tile_cols - 1
    grid = np.full((height, width), separator_color, dtype=int)

    idx = 0
    for tr in range(tile_rows):
        r0 = tr * (tile_h + 1)
        for tc in range(tile_cols):
            c0 = tc * (tile_w + 1)
            grid[r0 : r0 + tile_h, c0 : c0 + tile_w] = _paint_mask(
                masks[idx], tile_colors
            )
            idx += 1
    return grid


def _has_simple_transform_distractor(
    unpaired: np.ndarray, masks_without_unpaired: list[np.ndarray]
) -> bool:
    simple = [~unpaired, np.rot90(unpaired, 2)]
    existing = {_mask_key(mask) for mask in masks_without_unpaired}
    return any(_mask_key(mask) in existing for mask in simple)


def _gen_attempt() -> np.ndarray | None:
    layout = random.choice(_LAYOUTS)
    tile_h = random.choice(_TILE_HEIGHTS)
    tile_w = random.choice(_TILE_WIDTHS)
    if layout[0] * tile_h + layout[0] - 1 > 30:
        return None
    if layout[1] * tile_w + layout[1] - 1 > 30:
        return None

    separator_color, color_a, color_b = random.sample(_COLORS, 3)
    tile_colors = (color_a, color_b)
    tile_count = layout[0] * layout[1]
    pair_count = (tile_count - 1) // 2

    used_keys: set[bytes] = set()
    masks: list[np.ndarray] = []
    for _ in range(pair_count):
        mask = _random_mask(tile_h, tile_w, used_keys)
        if mask is None:
            return None
        partner = _mask_partner(mask)
        masks.extend([mask, partner])
        used_keys.add(_mask_key(mask))
        used_keys.add(_mask_key(partner))

    unpaired = _make_counterexample_unpaired(masks, used_keys)
    if unpaired is None:
        return None
    expected_mask = _mask_partner(unpaired)
    if _mask_key(expected_mask) in used_keys:
        return None
    if not _has_simple_transform_distractor(unpaired, masks):
        return None

    all_masks = masks + [unpaired]
    random.shuffle(all_masks)
    grid = _render_grid(
        all_masks,
        layout,
        (tile_h, tile_w),
        tile_colors,
        separator_color,
    )
    expected = _paint_mask(expected_mask, tile_colors)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(expected, _paint_mask(unpaired, tile_colors)):
        return None
    if np.array_equal(expected, _paint_mask(~unpaired, tile_colors)):
        return None
    if np.array_equal(expected, _paint_mask(np.rot90(unpaired, 2), tile_colors)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid composite-pairing tile grid")
