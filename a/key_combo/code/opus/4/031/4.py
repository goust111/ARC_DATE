import itertools
import random

import numpy as np


LEFT_WIDTH = 9
SEPARATOR_COLOR = 4
TILE_COLOR = 7
BG = 0
TILE_SIZE = 5
BAND_COLORS = (2, 3, 1)  # red, green, blue
TRANSFORM_BY_COLOR = {
    2: lambda tile: tile.copy(),
    3: lambda tile: np.fliplr(tile),
    1: lambda tile: np.flipud(tile),
}
_BAND_ORDER_BAG = []


def _row_majority_color(row: np.ndarray) -> int:
    counts = np.bincount(np.asarray(row, dtype=int), minlength=10)
    color = int(np.argmax(counts))
    top = int(counts[color])
    if top < 7 or int(np.count_nonzero(counts == top)) != 1:
        raise ValueError("left-side row does not have a clear band majority")
    return color


def _compress_runs(values: list[int]) -> list[int]:
    out = []
    for value in values:
        if not out or out[-1] != value:
            out.append(value)
    return out


def _find_separator_col(grid: np.ndarray) -> int:
    candidates = [
        int(c)
        for c in np.where(np.all(grid == SEPARATOR_COLOR, axis=0))[0]
        if c >= LEFT_WIDTH and grid.shape[1] - c - 1 >= TILE_SIZE
    ]
    if len(candidates) != 1:
        raise ValueError("expected exactly one full-height yellow separator column")
    return candidates[0]


def _extract_band_sequence(grid: np.ndarray) -> list[int]:
    sep = _find_separator_col(grid)
    left = np.asarray(grid[:, :sep], dtype=int)
    if left.shape[1] != LEFT_WIDTH:
        raise ValueError("unexpected left-region width")
    sequence = _compress_runs([_row_majority_color(row) for row in left])
    if len(sequence) != 3 or set(sequence) != set(BAND_COLORS):
        raise ValueError("expected exactly three compressed band colors")
    return sequence


def _extract_tile(grid: np.ndarray) -> np.ndarray:
    sep = _find_separator_col(grid)
    tile = np.asarray(grid[:TILE_SIZE, sep + 1 : sep + 1 + TILE_SIZE], dtype=int)
    if tile.shape != (TILE_SIZE, TILE_SIZE):
        raise ValueError("could not extract 5x5 tile")
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the left color bands and use their color order to choose tile flips on the right.

    Input structure:
    - A full-height yellow separator column splits the grid into a 9-column left band area and a
      5-column right area.
    - The left area is filled by three horizontal bands made from red, green, and blue, each
      appearing exactly once from top to bottom, with noisy jagged edges but a clear row-majority.
    - The top of the right area contains a 5x5 orange-on-black tile; the cells below it are black.

    Rule:
    - Find the separator, then read the left area row by row and compute each row's majority color.
    - Compress consecutive equal row-majorities to recover the three-band color sequence.
    - Interpret band colors as transform instructions for the right tile:
      red keeps the tile unchanged, green mirrors it left-right, and blue mirrors it up-down.
    - Apply the instruction for each compressed band in top-to-bottom order and stack the three
      transformed 5x5 tiles vertically.

    Key insight:
    - The irregular left bands are not output content; after row-majority compression they become a
      three-symbol program that determines which mirror transform each output tile block receives.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    sequence = _extract_band_sequence(g)
    tile = _extract_tile(g)
    blocks = [TRANSFORM_BY_COLOR[color](tile) for color in sequence]
    return np.vstack(blocks).astype(int)


def _next_band_order() -> tuple[int, int, int]:
    global _BAND_ORDER_BAG
    if not _BAND_ORDER_BAG:
        _BAND_ORDER_BAG = list(itertools.permutations(BAND_COLORS))
        random.shuffle(_BAND_ORDER_BAG)
    return tuple(_BAND_ORDER_BAG.pop())


def _pick_band_heights() -> tuple[int, int, int]:
    return tuple(random.choice(list(itertools.product((3, 4, 5), repeat=3))))


def _make_band_panel(order: tuple[int, int, int], heights: tuple[int, int, int]) -> np.ndarray:
    panel = np.zeros((sum(heights), LEFT_WIDTH), dtype=int)
    starts = []
    r0 = 0
    for color, height in zip(order, heights):
        panel[r0 : r0 + height, :] = color
        starts.append(r0)
        r0 += height

    for band_idx, (color, height) in enumerate(zip(order, heights)):
        base_r = starts[band_idx]
        for local_r in range(height):
            r = base_r + local_r
            edge_bias = local_r in {0, height - 1}
            noise_budget = random.choices((0, 1, 2), weights=(2, 5, 4) if edge_bias else (7, 3, 1))[0]
            if noise_budget == 0:
                continue

            candidates = []
            if band_idx > 0 and local_r <= 1:
                candidates.append(order[band_idx - 1])
            if band_idx < 2 and local_r >= height - 2:
                candidates.append(order[band_idx + 1])
            if not candidates:
                candidates = [c for c in BAND_COLORS if c != color]

            for _ in range(noise_budget):
                side = random.choice(("left", "right", "inner"))
                if side == "left":
                    c = random.choice((0, 1, 2))
                elif side == "right":
                    c = random.choice((6, 7, 8))
                else:
                    c = random.randrange(LEFT_WIDTH)
                panel[r, c] = random.choice(candidates)

            panel[r, random.randrange(LEFT_WIDTH)] = color
            panel[r, random.randrange(LEFT_WIDTH)] = color

    return panel


def _tile_variants(tile: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    original = np.asarray(tile, dtype=int)
    return original, np.fliplr(original), np.flipud(original)


def _is_good_tile(tile: np.ndarray) -> bool:
    if tile.shape != (TILE_SIZE, TILE_SIZE):
        return False
    orange = int(np.count_nonzero(tile == TILE_COLOR))
    if orange < 5 or orange > 14:
        return False
    original, hflip, vflip = _tile_variants(tile)
    if np.array_equal(original, hflip) or np.array_equal(original, vflip):
        return False
    if np.array_equal(hflip, vflip):
        return False
    if not np.any(tile[0] == TILE_COLOR) and not np.any(tile[:, 0] == TILE_COLOR):
        return False
    return True


def _make_tile() -> np.ndarray:
    while True:
        tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=int)
        orange_cells = random.randint(6, 12)
        coords = random.sample([(r, c) for r in range(TILE_SIZE) for c in range(TILE_SIZE)], orange_cells)
        for r, c in coords:
            tile[r, c] = TILE_COLOR
        if _is_good_tile(tile):
            return tile


def _validate_generated_input(grid: np.ndarray) -> None:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        raise ValueError("generated input must be a 2D ndarray")
    if max(grid.shape) > 30:
        raise ValueError("generated input exceeds ARC size limit")

    sep = _find_separator_col(grid)
    if sep != LEFT_WIDTH:
        raise ValueError("separator moved away from the intended reference column")
    if np.any(grid[:, sep + 1 + TILE_SIZE :] != BG):
        raise ValueError("unexpected content beyond the right tile area")

    sequence = _extract_band_sequence(grid)
    tile = _extract_tile(grid)
    if not _is_good_tile(tile):
        raise ValueError("tile does not distinguish the three transforms")

    expected = np.vstack([TRANSFORM_BY_COLOR[color](tile) for color in sequence])
    actual = main(grid)
    if not np.array_equal(actual, expected):
        raise ValueError("solver output does not match the encoded transform program")
    if np.unique(actual.reshape(3, TILE_SIZE, TILE_SIZE), axis=0).shape[0] != 3:
        raise ValueError("three output blocks must stay distinct")


def generate_input() -> np.ndarray:
    for _ in range(200):
        order = _next_band_order()
        heights = _pick_band_heights()
        height = sum(heights)

        grid = np.zeros((height, LEFT_WIDTH + 1 + TILE_SIZE), dtype=int)
        grid[:, :LEFT_WIDTH] = _make_band_panel(order, heights)
        grid[:, LEFT_WIDTH] = SEPARATOR_COLOR
        grid[:TILE_SIZE, LEFT_WIDTH + 1 :] = _make_tile()

        try:
            _validate_generated_input(grid)
        except ValueError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
