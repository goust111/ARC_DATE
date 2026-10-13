import numpy as np
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Return the single repeated tile from a grid made of two identical adjacent copies.

    Input structure
    - The grid is a rectangular pattern duplicated exactly once, either side-by-side
      or top-to-bottom.
    - The two copies occupy equal-sized halves along the duplication axis.

    Rule
    - Find the only axis whose two equal halves are identical.
    - Output the first half along that axis.

    Key insight
    - The output is not recolored or otherwise transformed; it is the source tile
      before the duplicate copy was appended.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    candidates: list[np.ndarray] = []
    if w % 2 == 0:
        left = g[:, : w // 2]
        right = g[:, w // 2 :]
        if np.array_equal(left, right):
            candidates.append(left)

    if h % 2 == 0:
        top = g[: h // 2, :]
        bottom = g[h // 2 :, :]
        if np.array_equal(top, bottom):
            candidates.append(top)

    assert len(candidates) == 1
    return candidates[0].copy()

# --------------- generator (o5x) ---------------

# Tile dimensions from the task spec.
_GEN_TILE_SIDE_RANGE = (2, 5)

# Palette choices: non-black ARC colors with 3-4 distinct colors per tile.
_GEN_COLOR_VALUES = range(1, 10)
_GEN_COLOR_BIN_COUNT = 10
_GEN_COLOR_COUNT_RANGE = (3, 4)

# Duplicate either top-to-bottom or side-by-side with equal weight.
_GEN_DUPLICATION_AXES = (0, 1)

# Retry limits for repairing tiles and building a valid input.
_GEN_REPAIR_RETRIES = 8
_GEN_ATTEMPT_RETRIES = 300


def _gen_has_equal_halves(tile: np.ndarray) -> bool:
    h, w = tile.shape
    if h % 2 == 0 and np.array_equal(tile[: h // 2, :], tile[h // 2 :, :]):
        return True
    if w % 2 == 0 and np.array_equal(tile[:, : w // 2], tile[:, w // 2 :]):
        return True
    return False


def _gen_matching_half_edits(tile: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return cell/mirror pairs that can break currently equal halves."""
    h, w = tile.shape
    edits = []
    if h % 2 == 0 and np.array_equal(tile[: h // 2, :], tile[h // 2 :, :]):
        for r in range(h // 2, h):
            for c in range(w):
                edits.append((r, c, r - h // 2, c))
    if w % 2 == 0 and np.array_equal(tile[:, : w // 2], tile[:, w // 2 :]):
        for r in range(h):
            for c in range(w // 2, w):
                edits.append((r, c, r, c - w // 2))
    return edits


def _gen_repair_equal_halves(tile: np.ndarray, colors: list[int]) -> bool:
    for _ in range(_GEN_REPAIR_RETRIES):
        if not _gen_has_equal_halves(tile):
            return True

        edits = _gen_matching_half_edits(tile)
        random.shuffle(edits)
        counts = np.bincount(tile.ravel(), minlength=_GEN_COLOR_BIN_COUNT)
        for r, c, rr, cc in edits:
            old = int(tile[r, c])
            options = [color for color in colors if color != int(tile[rr, cc])]
            random.shuffle(options)
            for color in options:
                if counts[old] <= 1:
                    continue
                tile[r, c] = color
                if not _gen_has_equal_halves(tile):
                    return True
                tile[r, c] = old

        if edits:
            r, c, rr, cc = edits[0]
            options = [color for color in colors if color != int(tile[rr, cc])]
            tile[r, c] = random.choice(options)

    return not _gen_has_equal_halves(tile)


def _gen_make_tile(h: int, w: int, colors: list[int]) -> np.ndarray | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    tile = np.empty((h, w), dtype=int)

    for (r, c), color in zip(cells, colors):
        tile[r, c] = color
    for r, c in cells[len(colors) :]:
        tile[r, c] = random.choice(colors)

    if not _gen_repair_equal_halves(tile, colors):
        return None
    return tile


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_GEN_TILE_SIDE_RANGE)
    w = random.randint(*_GEN_TILE_SIDE_RANGE)
    axis = random.choice(_GEN_DUPLICATION_AXES)
    color_count = random.randint(*_GEN_COLOR_COUNT_RANGE)
    colors = random.sample(_GEN_COLOR_VALUES, color_count)

    tile = _gen_make_tile(h, w, colors)
    if tile is None:
        return None

    out = tile.copy()
    inp = np.concatenate([out, out], axis=axis)

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPT_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 7b7f7511 input")
