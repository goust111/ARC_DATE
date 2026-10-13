import numpy as np

from arc_utils import Color, get_bg


DIAMOND_CELLS = ((0, 1), (1, 0), (1, 2), (2, 1))
DIAMOND_SET = set(DIAMOND_CELLS)
NEIGHBOR_DIRECTIONS = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)


def _bands(separator_indices: list[int], limit: int) -> list[tuple[int, int]]:
    bands = []
    start = 0
    for sep in separator_indices:
        assert start < sep
        bands.append((start, sep))
        start = sep + 1
    assert start < limit
    bands.append((start, limit))
    return bands


def _parse_board(g: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], dict[tuple[int, int], int]]:
    assert g.shape == (11, 11)
    bg = get_bg(g)
    assert bg == Color.ORANGE

    row_seps = [r for r in range(g.shape[0]) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg]
    col_seps = [c for c in range(g.shape[1]) if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg]
    assert len(row_seps) == 2 and len(col_seps) == 2
    assert {int(g[r, 0]) for r in row_seps} == {Color.PINK}
    assert {int(g[0, c]) for c in col_seps} == {Color.PINK}

    row_bands = _bands(row_seps, g.shape[0])
    col_bands = _bands(col_seps, g.shape[1])
    assert [r1 - r0 for r0, r1 in row_bands] == [3, 3, 3]
    assert [c1 - c0 for c0, c1 in col_bands] == [3, 3, 3]

    sprites: dict[tuple[int, int], int] = {}
    for br, (r0, r1) in enumerate(row_bands):
        for bc, (c0, c1) in enumerate(col_bands):
            patch = g[r0:r1, c0:c1]
            colors = set(map(int, patch.ravel())) - {bg}
            if not colors:
                continue
            assert len(colors) == 1
            color = colors.pop()
            coords = set(map(tuple, np.argwhere(patch == color)))
            assert coords == DIAMOND_SET
            sprites[(br, bc)] = color

    return bg, row_bands, col_bands, sprites


def _paint_diamond(
    out: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    board_pos: tuple[int, int],
    color: int,
    bg: int,
) -> None:
    r0, r1 = row_bands[board_pos[0]]
    c0, c1 = col_bands[board_pos[1]]
    patch = out[r0:r1, c0:c1]
    assert patch.shape == (3, 3)
    assert np.all(patch == bg)
    for rr, cc in DIAMOND_CELLS:
        patch[rr, cc] = int(color)


def _expected_added_sprites(red_arm_pos: tuple[int, int]) -> dict[tuple[int, int], int]:
    dr = red_arm_pos[0] - 1
    dc = red_arm_pos[1] - 1
    assert (dr, dc) in NEIGHBOR_DIRECTIONS
    direction_index = NEIGHBOR_DIRECTIONS.index((dr, dc))
    grey_dr, grey_dc = NEIGHBOR_DIRECTIONS[(direction_index + 3) % len(NEIGHBOR_DIRECTIONS)]
    teal_dr, teal_dc = NEIGHBOR_DIRECTIONS[(direction_index - 3) % len(NEIGHBOR_DIRECTIONS)]
    return {
        (1 + grey_dr, 1 + grey_dc): Color.GREY,
        (1 + teal_dr, 1 + teal_dc): Color.TEAL,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add the two missing colored arms of a centered diamond-sprite motif, or clear a completed motif to a blank square.

    Input structure
    An 11x11 board has orange background and pink separator bars forming nine 3x3 cells. Occupied cells contain a four-pixel diamond sprite; the center cell and one neighboring cell are red, and a completed board may also contain one grey and one teal sprite.

    Rule
    With only the two red sprites present, preserve the board and add grey and teal diamond sprites 135 degrees clockwise and counterclockwise from the non-center red arm. If those expected grey and teal sprites are already present, output an all-background square whose side length is the total number of sprite pixels.

    Key insight
    The non-center red sprite is an eight-neighbor direction from the center; the other two colored arms are fixed rotations of that direction.
    """
    g = np.asarray(grid, dtype=int)
    bg, row_bands, col_bands, sprites = _parse_board(g)

    red_positions = [pos for pos, color in sprites.items() if color == Color.RED]
    assert len(red_positions) == 2 and (1, 1) in red_positions
    red_arm_pos = [pos for pos in red_positions if pos != (1, 1)][0]
    expected = _expected_added_sprites(red_arm_pos)

    non_red_sprites = {pos: color for pos, color in sprites.items() if color != Color.RED}
    if non_red_sprites:
        assert len(sprites) == 4
        assert non_red_sprites == expected
        side = len(sprites) * len(DIAMOND_CELLS)
        return np.full((side, side), bg, dtype=int)

    out = g.copy()
    for pos, color in expected.items():
        assert pos not in sprites
        _paint_diamond(out, row_bands, col_bands, pos, color, bg)
    return out

# --------------- generator (att2) ---------------

GRID_SHAPE = (11, 11)
CELL_BANDS = ((0, 3), (4, 7), (8, 11))
SEPARATOR_INDICES = (3, 7)
CENTER_POS = (1, 1)
MODE_B_PROBABILITY = 0.25
MAX_GENERATION_ATTEMPTS = 300

def _gen_blank_board() -> tuple[np.ndarray, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    row_bands = CELL_BANDS
    col_bands = CELL_BANDS
    grid = np.full(GRID_SHAPE, Color.ORANGE, dtype=int)
    grid[list(SEPARATOR_INDICES), :] = Color.PINK
    grid[:, list(SEPARATOR_INDICES)] = Color.PINK
    return grid, row_bands, col_bands


def _gen_attempt() -> np.ndarray | None:
    grid, row_bands, col_bands = _gen_blank_board()
    direction = NEIGHBOR_DIRECTIONS[np.random.randint(len(NEIGHBOR_DIRECTIONS))]
    arm_pos = (CENTER_POS[0] + direction[0], CENTER_POS[1] + direction[1])
    expected = _expected_added_sprites(arm_pos)

    _paint_diamond(grid, row_bands, col_bands, CENTER_POS, Color.RED, Color.ORANGE)
    _paint_diamond(grid, row_bands, col_bands, arm_pos, Color.RED, Color.ORANGE)
    if np.random.random() < MODE_B_PROBABILITY:
        for pos, color in expected.items():
            _paint_diamond(grid, row_bands, col_bands, pos, color, Color.ORANGE)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            main(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate valid input")
