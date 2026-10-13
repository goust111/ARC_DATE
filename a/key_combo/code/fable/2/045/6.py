from __future__ import annotations

from collections import Counter
import random

import numpy as np


# Minimal helpers adapted from /home/lixinyu/snap/new/solvers/arc_utils.py so the
# file remains loadable via SourceFileLoader without external path assumptions.
BACKGROUND = 0
SEP_TO_OP = {
    2: "flip_lr",
    3: "flip_ud",
    4: "rot180",
    6: "identity",
}
SEP_COLORS = tuple(SEP_TO_OP.keys())
PATTERN_COLORS = (1, 5, 7, 8, 9)
GEN_RETRIES = 500
MOTIF_LIBRARY = (
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> np.ndarray:
    h, w = patch.shape
    r1 = int(r0) + h
    c1 = int(c0) + w
    view = dst[int(r0) : r1, int(c0) : c1]
    if transparent is None:
        view[:, :] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]
    return dst


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        dirs = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _apply_op(tile: np.ndarray, op_name: str) -> np.ndarray:
    if op_name == "identity":
        return tile.copy()
    if op_name == "flip_lr":
        return np.fliplr(tile).copy()
    if op_name == "flip_ud":
        return np.flipud(tile).copy()
    if op_name == "rot180":
        return np.rot90(tile, 2).copy()
    raise ValueError(f"unknown op: {op_name}")


def _parse_chain(grid: np.ndarray) -> tuple[int, list[np.ndarray], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    tile_size = int(g.shape[0])
    width = int(g.shape[1])
    assert tile_size >= 3
    assert (width + 1) % (tile_size + 1) == 0, "width must match a tile-chain layout"
    n_tiles = (width + 1) // (tile_size + 1)
    assert n_tiles >= 2

    bg = get_bg(g)
    tiles: list[np.ndarray] = []
    separators: list[int] = []
    for idx in range(n_tiles):
        c0 = idx * (tile_size + 1)
        tiles.append(g[:, c0 : c0 + tile_size].copy())
        if idx < n_tiles - 1:
            sep_col = c0 + tile_size
            col = g[:, sep_col]
            assert np.all(col == col[0]), "separator must be a solid color column"
            sep_color = int(col[0])
            assert sep_color in SEP_TO_OP and sep_color != bg
            separators.append(sep_color)

    assert np.any(tiles[0] != bg), "the first tile must contain the seed pattern"
    for tile in tiles[1:]:
        assert np.all(tile == bg), "all non-leftmost tiles must start empty"
    return bg, tiles, separators


def _render_from_states(states: list[np.ndarray], separators: list[int]) -> np.ndarray:
    tile_size = int(states[0].shape[0])
    width = len(states) * tile_size + len(separators)
    out = np.full((tile_size, width), BACKGROUND, dtype=int)
    for idx, tile in enumerate(states):
        paste(out, tile, 0, idx * (tile_size + 1), transparent=BACKGROUND)
        if idx < len(separators):
            out[:, idx * (tile_size + 1) + tile_size] = int(separators[idx])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the first tile through the separator-coded pipeline and fill every empty tile with the resulting stage.

    Input structure:
    - The grid is a single horizontal chain of equally sized square tiles.
    - Neighboring tiles are separated by one-cell-wide solid vertical separator columns.
    - Only the leftmost tile contains a colored pattern; every other tile is background-only.

    Rule:
    - Read the separator columns from left to right.
    - Each separator color encodes one fixed transform on a tile: left-right mirror, up-down mirror,
      180-degree rotation, or identity copy.
    - Starting from the leftmost tile, apply each separator's transform to the current tile state and
      write the result into the tile immediately to its right. Keep composing stage by stage across the chain.
    - Preserve every separator column unchanged.

    Key insight:
    - The separators are instructions, not dividers: each tile stores the intermediate result after all
      transforms to its left have already been composed.
    """
    bg, tiles, separators = _parse_chain(grid)
    states = [tiles[0]]
    current = tiles[0]
    for sep_color in separators:
        current = _apply_op(current, SEP_TO_OP[sep_color])
        states.append(current)
    out = _render_from_states(states, separators)
    out[out == BACKGROUND] = bg
    return out


def _all_op_images(tile: np.ndarray) -> dict[str, np.ndarray]:
    return {name: _apply_op(tile, name) for name in SEP_TO_OP.values()}


def _has_separator_like_stripe(tile: np.ndarray) -> bool:
    for c in range(tile.shape[1]):
        col = tile[:, c]
        if col[0] != BACKGROUND and np.all(col == col[0]):
            return True
    return False


def _source_is_valid(tile: np.ndarray) -> bool:
    box = bbox(tile != BACKGROUND)
    if box is None:
        return False
    cropped = crop_bbox(tile, box)
    if cropped.shape[0] < 3 or cropped.shape[1] < 3:
        return False
    if len(set(map(int, tile[tile != BACKGROUND].flat))) < 2:
        return False
    if int(np.count_nonzero(tile)) < tile.shape[0] + 1:
        return False
    if _has_separator_like_stripe(tile):
        return False

    comps = components_mask(tile != BACKGROUND, connectivity=4)
    if max(comp.shape[0] for comp in comps) < 3:
        return False

    images = _all_op_images(tile)
    unique = {img.tobytes() for img in images.values()}
    if len(unique) != 4:
        return False
    return True


def _make_source_tile(tile_size: int) -> np.ndarray | None:
    for _ in range(200):
        tile = np.zeros((tile_size, tile_size), dtype=int)
        occupancy = np.zeros_like(tile, dtype=bool)
        palette = random.sample(PATTERN_COLORS, random.randint(2, min(4, len(PATTERN_COLORS))))
        motif_count = random.randint(2, 4 if tile_size >= 5 else 3)

        ok = True
        for _ in range(motif_count):
            motif = random.choice(MOTIF_LIBRARY)
            mh, mw = motif.shape
            if mh > tile_size or mw > tile_size:
                ok = False
                break
            placed = False
            for _ in range(40):
                r0 = random.randint(0, tile_size - mh)
                c0 = random.randint(0, tile_size - mw)
                motif_mask = motif.astype(bool)
                if np.any(occupancy[r0 : r0 + mh, c0 : c0 + mw] & motif_mask):
                    continue
                color = random.choice(palette)
                patch = np.where(motif_mask, color, BACKGROUND).astype(int)
                paste(tile, patch, r0, c0, transparent=BACKGROUND)
                occupancy[r0 : r0 + mh, c0 : c0 + mw] |= motif_mask
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        accent_count = random.randint(1, max(1, tile_size - 3))
        empty_coords = list(zip(*np.where(tile == BACKGROUND)))
        random.shuffle(empty_coords)
        for r, c in empty_coords[:accent_count]:
            tile[int(r), int(c)] = random.choice(palette)

        if _source_is_valid(tile):
            return tile
    return None


def _build_separator_sequence(n_tiles: int) -> list[int]:
    separators = list(SEP_COLORS)
    while len(separators) < n_tiles - 1:
        separators.append(random.choice(SEP_COLORS))
    random.shuffle(separators)
    return separators


def _states_from_source(source: np.ndarray, separators: list[int]) -> list[np.ndarray]:
    states = [source]
    current = source
    for sep_color in separators:
        current = _apply_op(current, SEP_TO_OP[sep_color])
        states.append(current)
    return states


def _shortcut_states(source: np.ndarray, separators: list[int]) -> list[np.ndarray]:
    states = [source]
    for sep_color in separators:
        states.append(_apply_op(source, SEP_TO_OP[sep_color]))
    return states


def _semantic_checks(source: np.ndarray, separators: list[int]) -> bool:
    states = _states_from_source(source, separators)
    shortcut = _shortcut_states(source, separators)

    # Reject instances where "every separator acts directly on the first tile" would
    # accidentally match the intended chained composition.
    diverged_steps = sum(
        not np.array_equal(states[idx], shortcut[idx])
        for idx in range(2, len(states))
    )
    if diverged_steps < 2:
        return False

    # Every adjacent tile pair must identify exactly one transform.
    op_names = tuple(SEP_TO_OP.values())
    for idx, sep_color in enumerate(separators):
        left = states[idx]
        right = states[idx + 1]
        matches = [name for name in op_names if np.array_equal(_apply_op(left, name), right)]
        if matches != [SEP_TO_OP[sep_color]]:
            return False

    # Use every opcode at least once so the mapping is always visible in one example.
    if set(separators) != set(SEP_COLORS):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_tiles = random.choice([5, 5, 6])
    tile_size = random.choice([4, 5]) if n_tiles == 5 else 4
    source = _make_source_tile(tile_size)
    if source is None:
        return None

    separators = _build_separator_sequence(n_tiles)
    if not _semantic_checks(source, separators):
        return None

    width = n_tiles * tile_size + (n_tiles - 1)
    grid = np.full((tile_size, width), BACKGROUND, dtype=int)
    paste(grid, source, 0, 0, transparent=BACKGROUND)
    for idx, sep_color in enumerate(separators):
        grid[:, (idx + 1) * tile_size + idx] = int(sep_color)

    # Oracle check: solver output must equal the independently rendered pipeline.
    expected = _render_from_states(_states_from_source(source, separators), separators)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid separator-pipeline input")
