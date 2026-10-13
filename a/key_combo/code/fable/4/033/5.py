from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
TILE_SIZE = 3
MACRO_SIZE = 3
GRID_SIZE = 11
SEPARATORS = (3, 7)
SPRITE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MODES = ("horizontal", "vertical", "quarter_turn")
SLOTS = tuple(np.ndindex(MACRO_SIZE, MACRO_SIZE))
CORNERS = ((0, 0), (0, 2), (2, 0), (2, 2))


def _bands(separators: list[int], size: int) -> list[tuple[int, int]]:
    """Return half-open non-separator bands, using canonical (row, col) axes."""
    starts = [0] + [s + 1 for s in separators]
    ends = separators + [size]
    bands = list(zip(starts, ends))
    assert all(lo < hi for lo, hi in bands)
    return bands


def _transform(mask: np.ndarray, mode: str) -> np.ndarray:
    if mode == "horizontal":
        return np.flipud(mask)
    if mode == "vertical":
        return np.fliplr(mask)
    if mode == "quarter_turn":
        return np.rot90(mask)
    raise ValueError(f"unknown symmetry mode: {mode}")


def _is_symmetric(mask: np.ndarray, mode: str) -> bool:
    """The empty tile is a slot, not a symmetric sprite."""
    return bool(mask.any()) and np.array_equal(mask, _transform(mask, mode))


def _tile_color(tile: np.ndarray) -> int | None:
    colors = set(map(int, tile[tile != BLACK]))
    if not colors:
        return None
    assert len(colors) == 1, "each sprite must have exactly one visible color"
    color = colors.pop()
    assert color != GREY, "grey is reserved for separator lines"
    return color


def _repair_for_mode(tile: np.ndarray, mode: str) -> tuple[tuple[int, int], np.ndarray] | None:
    """Return the unique add-only repair for one established symmetry mode."""
    color = _tile_color(tile)
    if color is None:
        return None

    mask = tile != BLACK
    if _is_symmetric(mask, mode):
        return None

    additions: list[tuple[tuple[int, int], np.ndarray]] = []
    for r, c in np.argwhere(~mask):
        repaired = tile.copy()
        repaired[int(r), int(c)] = color
        if _is_symmetric(repaired != BLACK, mode):
            additions.append(((int(r), int(c)), repaired))

    # A deletion is a competing reading only if a non-empty sprite remains.
    deletions = 0
    for r, c in np.argwhere(mask):
        reduced = mask.copy()
        reduced[int(r), int(c)] = False
        if _is_symmetric(reduced, mode):
            deletions += 1

    if len(additions) != 1 or deletions != 0:
        return None
    return additions[0]


def _parse_tiles(g: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict[tuple[int, int], np.ndarray]]:
    assert g.ndim == 2 and g.shape[0] == g.shape[1]
    h, w = g.shape
    row_seps = [r for r in range(h) if np.all(g[r, :] == GREY)]
    col_seps = [c for c in range(w) if np.all(g[:, c] == GREY)]
    assert len(row_seps) == len(col_seps) == 2

    row_bands = _bands(row_seps, h)
    col_bands = _bands(col_seps, w)
    assert len(row_bands) == len(col_bands) == MACRO_SIZE
    assert {hi - lo for lo, hi in row_bands} == {TILE_SIZE}
    assert {hi - lo for lo, hi in col_bands} == {TILE_SIZE}

    non_separator = np.ones_like(g, dtype=bool)
    non_separator[row_seps, :] = False
    non_separator[:, col_seps] = False
    assert not np.any(g[non_separator] == GREY)

    tiles: dict[tuple[int, int], np.ndarray] = {}
    for br, (r0, r1) in enumerate(row_bands):
        for bc, (c0, c1) in enumerate(col_bands):
            tiles[(br, bc)] = g[r0:r1, c0:c1].copy()
    return row_bands, col_bands, tiles


def main(grid: np.ndarray) -> np.ndarray:
    """Repair the uniquely symmetry-broken sprite and copy it to the slot encoded by its missing cell.

    Input structure:
    - A square canvas is divided by two full grey rows and columns into a 3x3
      array of 3x3 tiles. Non-empty tiles hold monochrome sprites on black.
    - The intact sprites establish one exact axial-mirror or quarter-turn
      symmetry; exactly one sprite is missing one colored cell under that mode.

    Rule:
    - Find the symmetry mode with exactly one broken non-empty tile and a unique
      add-only repair. Fill its missing cell with the sprite color.
    - Use the missing cell's local (row, column) as the destination macro slot,
      clear every non-separator cell, and stamp the repaired 3x3 sprite there.

    Key insight:
    - The single black defect simultaneously selects the source, supplies the
      symmetry completion, and encodes the repaired sprite's destination.
    """
    g = np.asarray(grid, dtype=int)
    row_bands, col_bands, tiles = _parse_tiles(g)

    occupied = [slot for slot, tile in tiles.items() if _tile_color(tile) is not None]
    assert len(occupied) >= 2, "expected several sprites"

    candidates: list[tuple[tuple[int, int], tuple[int, int], np.ndarray]] = []
    for mode in MODES:
        broken = [slot for slot in occupied if not _is_symmetric(tiles[slot] != BLACK, mode)]
        if len(broken) != 1:
            continue
        source = broken[0]
        repair = _repair_for_mode(tiles[source], mode)
        if repair is not None:
            defect, repaired = repair
            candidates.append((source, defect, repaired))

    # Different transformations may certify the same visible repair.  Only the
    # source, missing coordinate, and repaired pixels must be uniquely determined.
    unique: dict[tuple[tuple[int, int], tuple[int, int], bytes], tuple[tuple[int, int], tuple[int, int], np.ndarray]] = {}
    for source, defect, repaired in candidates:
        key = (source, defect, repaired.tobytes())
        unique[key] = (source, defect, repaired)
    assert len(unique) == 1, "the broken sprite and its add-only repair must be unique"

    _source, (dst_br, dst_bc), repaired = next(iter(unique.values()))
    assert 0 <= dst_br < MACRO_SIZE and 0 <= dst_bc < MACRO_SIZE

    out = np.full_like(g, BLACK)
    row_seps = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    col_seps = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    out[row_seps, :] = GREY
    out[:, col_seps] = GREY
    r0, r1 = row_bands[dst_br]
    c0, c1 = col_bands[dst_bc]
    out[r0:r1, c0:c1] = repaired
    return out


def _empty_grid() -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), BLACK, dtype=int)
    grid[list(SEPARATORS), :] = GREY
    grid[:, list(SEPARATORS)] = GREY
    return grid


def _tile_slice(slot: tuple[int, int]) -> tuple[slice, slice]:
    br, bc = slot
    return slice(4 * br, 4 * br + TILE_SIZE), slice(4 * bc, 4 * bc + TILE_SIZE)


def _mask_options(mode: str) -> list[np.ndarray]:
    options: list[np.ndarray] = []
    for bits in range(1, 1 << (TILE_SIZE * TILE_SIZE)):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        area = int(mask.sum())
        if 1 <= area <= 8 and _is_symmetric(mask, mode):
            options.append(mask)
    return options


MASK_OPTIONS = {mode: _mask_options(mode) for mode in MODES}


def _source_masks(mode: str, defect: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Construct the smallest unambiguous add-only defect for the chosen mode."""
    repaired = np.zeros((3, 3), dtype=bool)
    r, c = defect
    if mode == "horizontal":
        repaired[r, c] = True
        repaired[2 - r, c] = True
    elif mode == "vertical":
        repaired[r, c] = True
        repaired[r, 2 - c] = True
    else:
        repaired[0, 0] = repaired[0, 2] = True
        repaired[2, 0] = repaired[2, 2] = True
        repaired[1, 1] = True
    damaged = repaired.copy()
    damaged[r, c] = False
    return damaged, repaired


def _mandatory_distractor(mode: str, source_area: int) -> np.ndarray:
    """Match source area while also ruling out a color/size shortcut."""
    mask = np.zeros((3, 3), dtype=bool)
    if mode == "horizontal":
        mask[1, random.choice((0, 2))] = True
    elif mode == "vertical":
        mask[random.choice((0, 2)), 1] = True
    else:
        mask[0, 1] = mask[1, 0] = mask[1, 2] = mask[2, 1] = True
    assert int(mask.sum()) == source_area and _is_symmetric(mask, mode)
    return mask


def _render(mask: np.ndarray, color: int) -> np.ndarray:
    tile = np.full((3, 3), BLACK, dtype=int)
    tile[mask] = int(color)
    return tile


def _validate_generated(
    inp: np.ndarray,
    expected: np.ndarray,
    *,
    mode: str,
    source: tuple[int, int],
    destination: tuple[int, int],
    damaged: np.ndarray,
    repaired: np.ndarray,
    source_color: int,
    occupied: list[tuple[int, int]],
) -> bool:
    """Check puzzle semantics and anti-shortcut properties, not only oracle closure."""
    try:
        assert inp.shape == expected.shape == (GRID_SIZE, GRID_SIZE)
        assert source != destination and destination not in occupied
        assert len(occupied) >= 4

        _rows, _cols, tiles = _parse_tiles(inp)
        assert np.array_equal(tiles[source] != BLACK, damaged)
        assert all(_is_symmetric(tiles[s] != BLACK, mode) for s in occupied if s != source)
        assert not _is_symmetric(damaged, mode)

        source_tile = tiles[source]
        repair = _repair_for_mode(source_tile, mode)
        assert repair is not None
        defect, repaired_tile = repair
        assert defect == destination
        assert np.array_equal(repaired_tile != BLACK, repaired)
        assert int(repaired_tile[defect]) == source_color

        # The outlier is not identifiable by color or occupied area.
        same_color_area = [
            s for s in occupied if s != source
            and _tile_color(tiles[s]) == source_color
            and int(np.count_nonzero(tiles[s])) == int(damaged.sum())
        ]
        assert same_color_area

        actual = main(inp)
        assert np.array_equal(actual, expected)
        assert not np.array_equal(actual, inp)

        # Omitting repair, omitting address decoding, or omitting erasure each
        # gives a visibly different result.
        wrong_unrepaired = _empty_grid()
        wrong_unrepaired[_tile_slice(destination)] = source_tile
        assert not np.array_equal(wrong_unrepaired, expected)
        wrong_unmoved = _empty_grid()
        wrong_unmoved[_tile_slice(source)] = repaired_tile
        assert not np.array_equal(wrong_unmoved, expected)
        wrong_unerased = inp.copy()
        wrong_unerased[_tile_slice(destination)] = repaired_tile
        assert not np.array_equal(wrong_unerased, expected)

        # Only the repaired sprite survives outside the separators.
        non_sep = np.ones(expected.shape, dtype=bool)
        non_sep[list(SEPARATORS), :] = False
        non_sep[:, list(SEPARATORS)] = False
        assert int(np.count_nonzero(expected[non_sep])) == int(repaired.sum())
        return True
    except (AssertionError, ValueError):
        return False


_mode_cycle: list[str] = []


def _next_mode() -> str:
    """Cycle branches so every three consecutive generated examples cover all modes."""
    global _mode_cycle
    if not _mode_cycle:
        _mode_cycle = list(MODES)
        random.shuffle(_mode_cycle)
    return _mode_cycle.pop()


def _gen_attempt(mode: str) -> np.ndarray | None:
    defect = random.choice(CORNERS)
    damaged, repaired = _source_masks(mode, defect)
    assert _is_symmetric(repaired, mode) and not _is_symmetric(damaged, mode)

    destination = defect
    source = random.choice([slot for slot in SLOTS if slot != destination])
    sprite_count = random.randint(4, 7)
    distractor_slots = random.sample(
        [slot for slot in SLOTS if slot not in (source, destination)],
        sprite_count - 1,
    )
    occupied = [source, *distractor_slots]

    source_color = random.choice(SPRITE_COLORS)
    source_tile = _render(damaged, source_color)
    tiles: dict[tuple[int, int], np.ndarray] = {source: source_tile}

    mandatory = _mandatory_distractor(mode, int(damaged.sum()))
    tiles[distractor_slots[0]] = _render(mandatory, source_color)

    forbidden = {repaired.tobytes(), mandatory.tobytes()}
    for slot in distractor_slots[1:]:
        options = [mask for mask in MASK_OPTIONS[mode] if mask.tobytes() not in forbidden]
        mask = random.choice(options).copy()
        color = random.choice(SPRITE_COLORS)
        tiles[slot] = _render(mask, color)

    inp = _empty_grid()
    for slot, tile in tiles.items():
        inp[_tile_slice(slot)] = tile

    repaired_tile = _render(repaired, source_color)
    expected = _empty_grid()
    expected[_tile_slice(destination)] = repaired_tile

    if not _validate_generated(
        inp,
        expected,
        mode=mode,
        source=source,
        destination=destination,
        damaged=damaged,
        repaired=repaired,
        source_color=source_color,
        occupied=occupied,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a diverse valid 11x11 instance with oracle and semantic checks."""
    mode = _next_mode()
    for _ in range(300):
        result = _gen_attempt(mode)
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate a valid symmetry-address sample for {mode}")
