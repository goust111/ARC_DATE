"""ARC task: use a two-color mask to dispatch matching bordered pattern tiles."""

from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as arrays of (row, column) coordinates."""
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[np.ndarray] = []
    height, width = mask.shape
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue = deque([(sr, sc)])
        cells = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _crop_component(grid: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    box = (int(r0), int(c0), int(r1), int(c1))
    return grid[box[0] : box[2] + 1, box[1] : box[3] + 1].copy(), box


def _border_color(patch: np.ndarray) -> int | None:
    if min(patch.shape) < 3:
        return None
    border = np.concatenate((patch[0], patch[-1], patch[1:-1, 0], patch[1:-1, -1]))
    values = np.unique(border)
    return int(values[0]) if len(values) == 1 and int(values[0]) != BG else None


def _parse(grid: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size, "expected a non-empty 2D grid"
    assert np.all((0 <= g) & (g <= 9)), "expected ARC palette colors"
    components = [_crop_component(g, cells)[0] for cells in _components(g != BG)]

    mask_candidates = [
        patch
        for patch in components
        if np.any(patch == BG) and len(set(map(int, patch[patch != BG]))) == 2
    ]
    assert len(mask_candidates) == 1, "expected one irregular two-color mask"
    mask = mask_candidates[0]
    mask_colors = set(map(int, mask[mask != BG]))

    tile_candidates: list[tuple[np.ndarray, int]] = []
    for patch in components:
        key = _border_color(patch)
        if key is not None and np.all(patch != BG) and np.any(patch[1:-1, 1:-1] != key):
            tile_candidates.append((patch, key))

    shape_counts: dict[tuple[int, int], int] = {}
    for patch, _key in tile_candidates:
        shape_counts[patch.shape] = shape_counts.get(patch.shape, 0) + 1
    tile_shapes = [shape for shape, count in shape_counts.items() if count >= 3]
    assert len(tile_shapes) == 1, "expected one family of at least three same-size pattern tiles"
    tile_shape = tile_shapes[0]
    family = [(patch, key) for patch, key in tile_candidates if patch.shape == tile_shape]
    assert len({key for _patch, key in family}) == len(family), "tile border colors must be unique"

    keyed_tiles = {key: patch for patch, key in family if key in mask_colors}
    assert set(keyed_tiles) == mask_colors, "each mask color must name exactly one tile"
    assert len(family) > len(keyed_tiles), "expected at least one distractor tile"

    token_candidates = [
        patch
        for patch in components
        if patch is not mask and len(np.unique(patch)) == 1 and int(patch.flat[0]) != BG
    ]
    assert len(token_candidates) == 1, "expected one solid monochrome background token"
    token_color = int(token_candidates[0].flat[0])
    return mask, keyed_tiles, token_color


def main(grid: np.ndarray) -> np.ndarray:
    """Expand a two-color mask, stamping the pattern tile named by each foreground color.

    Input structure:
    - A black canvas contains one irregular, 4-connected two-color mask sprite.
    - It also contains at least three separated, equally sized full rectangular pattern
      tiles with monochrome borders, plus one separated solid monochrome rectangle.
    - Exactly two tile-border colors occur in the mask; all other bordered tiles are decoys.

    Rule:
    - Use the mask's own bounding-box row and column coordinates as the block layout.
    - Replace each colored mask cell by the complete tile whose border has that color.
    - Replace each black cell inside the mask bounding box by a solid block in the
      monochrome token's color. Tile and token positions and decoy tiles are ignored.

    Key insight:
    - A mask color is a pointer to a bordered tile, while the mask geometry supplies
      the Kronecker-style enlarged layout; both clues are necessary to form the output.
    """
    mask, keyed_tiles, token_color = _parse(grid)
    tile_h, tile_w = next(iter(keyed_tiles.values())).shape
    out = np.full((mask.shape[0] * tile_h, mask.shape[1] * tile_w), token_color, dtype=int)
    for r, c in np.argwhere(mask != BG):
        r, c = int(r), int(c)
        tile = keyed_tiles[int(mask[r, c])]
        out[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tile
    assert max(out.shape) <= 30, "output exceeds ARC's 30x30 limit"
    return out


def _connected(mask: np.ndarray) -> bool:
    parts = _components(mask)
    return len(parts) == 1


def _make_mask(height: int, width: int, colors: tuple[int, int]) -> np.ndarray:
    occupied = np.zeros((height, width), dtype=bool)
    cross_r = random.randrange(height)
    cross_c = random.randrange(width)
    occupied[cross_r, :] = True
    occupied[:, cross_c] = True
    maximum = height * width - 1
    target = random.randint(int(occupied.sum()), max(int(occupied.sum()), maximum))
    while int(occupied.sum()) < target:
        frontier = []
        for r, c in np.argwhere(occupied):
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and not occupied[nr, nc]:
                    frontier.append((nr, nc))
        occupied[random.choice(frontier)] = True
    assert occupied.sum() < occupied.size and _connected(occupied)
    patch = np.zeros((height, width), dtype=int)
    cells = np.argwhere(occupied)
    cells = cells[np.random.permutation(len(cells))]
    split = random.randint(1, len(cells) - 1)
    patch[cells[:split, 0], cells[:split, 1]] = colors[0]
    patch[cells[split:, 0], cells[split:, 1]] = colors[1]
    return patch


def _make_tile(height: int, width: int, border_color: int, palette: list[int]) -> np.ndarray:
    tile = np.full((height, width), border_color, dtype=int)
    interior = tile[1:-1, 1:-1]
    choices = [color for color in palette if color != border_color]
    interior[:] = np.random.choice(choices, size=interior.shape)
    # Every tile visibly has an interior pattern rather than being a solid token.
    interior[random.randrange(interior.shape[0]), random.randrange(interior.shape[1])] = random.choice(choices)
    return tile


def _place_objects(objects: list[np.ndarray], height: int, width: int) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    for patch in sorted(objects, key=lambda x: x.size, reverse=True):
        ph, pw = patch.shape
        choices = [(r, c) for r in range(height - ph + 1) for c in range(width - pw + 1)]
        random.shuffle(choices)
        placed = False
        for r, c in choices:
            r0, r1 = max(0, r - 1), min(height, r + ph + 1)
            c0, c1 = max(0, c - 1), min(width, c + pw + 1)
            if blocked[r0:r1, c0:c1].any():
                continue
            grid[r : r + ph, c : c + pw] = patch
            blocked[r : r + ph, c : c + pw] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _semantic_validation(grid: np.ndarray, expected_mask: np.ndarray, expected_tiles: dict[int, np.ndarray], token: int) -> None:
    parsed_mask, parsed_tiles, parsed_token = _parse(grid)
    assert np.array_equal(parsed_mask, expected_mask)
    assert parsed_token == token and set(parsed_tiles) == set(expected_tiles)
    th, tw = next(iter(expected_tiles.values())).shape
    expected = np.full((expected_mask.shape[0] * th, expected_mask.shape[1] * tw), token, dtype=int)
    for r in range(expected_mask.shape[0]):
        for c in range(expected_mask.shape[1]):
            color = int(expected_mask[r, c])
            if color != BG:
                expected[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = expected_tiles[color]
    assert np.array_equal(main(grid), expected)
    # Non-degeneracy: both pointers, the mask background, and distinct named tiles matter.
    assert set(map(int, expected_mask[expected_mask != BG])) == set(expected_tiles)
    assert np.any(expected_mask == BG)
    assert not np.array_equal(*expected_tiles.values())


def generate_input() -> np.ndarray:
    """Generate diverse, non-degenerate examples satisfying every visible role constraint."""
    for _attempt in range(300):
        tile_h, tile_w = random.randint(3, 5), random.randint(3, 5)
        mask_h = random.randint(2, min(5, 30 // tile_h))
        mask_w = random.randint(2, min(5, 30 // tile_w))
        distractor_count = random.randint(1, 3)
        role_colors = random.sample(PALETTE, 2 + distractor_count + 1)
        mask_colors = (role_colors[0], role_colors[1])
        border_colors = role_colors[: 2 + distractor_count]
        token_color = role_colors[-1]

        mask = _make_mask(mask_h, mask_w, mask_colors)
        tiles = [_make_tile(tile_h, tile_w, color, list(PALETTE)) for color in border_colors]
        # Make every pattern distinct, so pointer selection cannot collapse to one stamp.
        if len({tile.tobytes() for tile in tiles}) != len(tiles):
            continue
        token_shapes = [(1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 2)]
        token_shape = random.choice([shape for shape in token_shapes if shape != (tile_h, tile_w)])
        token = np.full(token_shape, token_color, dtype=int)

        objects = [mask, token, *tiles]
        random.shuffle(objects)
        canvas_h, canvas_w = random.randint(20, 30), random.randint(20, 30)
        grid = _place_objects(objects, canvas_h, canvas_w)
        if grid is None:
            continue
        expected_tiles = {color: tile for color, tile in zip(border_colors[:2], tiles[:2])}
        try:
            _semantic_validation(grid, mask, expected_tiles, token_color)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid mask-and-tile input")
