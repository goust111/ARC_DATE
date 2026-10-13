import random

import numpy as np


BLACK = 0
GREY = 5
PATH_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRY_LIMIT = 500


def _read_palette_and_controls(g: np.ndarray) -> tuple[list[int], dict[int, tuple[int, int]]]:
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = map(int, g.shape)
    assert height >= 6 and width >= 6, "grid is too small for palette-ordered paths"

    palette: list[int] = []
    r = 0
    while r < height and int(g[r, 0]) != BLACK:
        color = int(g[r, 0])
        assert color != GREY, "grey is reserved for blockers, not palette entries"
        assert color not in palette, "palette colors must be distinct"
        palette.append(color)
        r += 1

    assert len(palette) >= 2, "expected at least two palette colors"
    assert r < height - 1, "palette must not run into the control row"
    assert np.all(g[r:, 0] == BLACK), "left column below the palette must be black"

    present = set(map(int, np.unique(g)))
    allowed = {BLACK, GREY, *palette}
    assert present <= allowed, "unexpected color outside the palette, black, and grey blockers"
    assert GREY not in set(map(int, g[-1, :])), "grey blockers must not be on the control row"

    starts: dict[int, tuple[int, int]] = {}
    for i, color in enumerate(palette):
        coords = np.argwhere(g == color)
        bottom_cols = [int(c) for rr, c in coords if int(rr) == height - 1]
        assert len(bottom_cols) == 1, "each palette color needs exactly one bottom control pixel"
        control = (height - 1, bottom_cols[0])
        assert control[1] > 0, "control pixels must be outside the palette column"
        expected = {(i, 0), control}
        assert {tuple(map(int, xy)) for xy in coords.tolist()} == expected, (
            "input may only contain the palette pixel and one control pixel for each path color"
        )
        starts[color] = control

    return palette, starts


def _classify_cell(
    pos: tuple[int, int] | None,
    value: int | None,
    previous_path: set[tuple[int, int]],
    current_path: set[tuple[int, int]],
) -> str:
    if pos is None:
        return "out"
    if value == BLACK:
        return "black"
    if value == GREY:
        return "grey"
    if pos in previous_path:
        return "previous_path"
    if pos in current_path:
        return "current_path"
    return "other_nonblack"


def _simulate_paths(
    g: np.ndarray,
    palette: list[int],
    starts: dict[int, tuple[int, int]],
    *,
    order: list[int] | None = None,
    use_current_state: bool = True,
    record_events: bool = False,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    height, width = map(int, g.shape)
    if order is None:
        order = list(palette)
    assert set(order) == set(palette) and len(order) == len(palette), "path order must use every palette color once"

    out = g.copy()
    previous_path: set[tuple[int, int]] = set()
    events: list[dict[str, object]] = []

    for color in order:
        r, c = starts[color]
        current_path = {(int(r), int(c))}

        while r > 0:
            sense = out if use_current_state else g
            above_pos = (r - 1, c)
            above_value = int(sense[above_pos])

            if above_value == BLACK:
                r -= 1
                out[r, c] = int(color)
                current_path.add((r, c))
                continue

            above_kind = _classify_cell(above_pos, above_value, previous_path, current_path)
            right_pos = None if c + 1 >= width else (r, c + 1)
            right_value = None if right_pos is None else int(sense[right_pos])
            right_kind = _classify_cell(right_pos, right_value, previous_path, current_path)

            if right_pos is None or right_value != BLACK:
                if record_events:
                    events.append(
                        {
                            "color": int(color),
                            "above_kind": above_kind,
                            "right_kind": right_kind,
                            "terminates": True,
                            "row": int(r),
                            "col": int(c),
                        }
                    )
                break

            c += 1
            out[r, c] = int(color)
            current_path.add((r, c))
            if record_events:
                events.append(
                    {
                        "color": int(color),
                        "above_kind": above_kind,
                        "right_kind": "black",
                        "terminates": False,
                        "row": int(r),
                        "col": int(c),
                    }
                )

        previous_path.update(current_path)

    return out, events


def main(grid: np.ndarray) -> np.ndarray:
    """Grow bottom control pixels upward in the order given by the left palette.

    Input structure:
    A black grid has a contiguous top-to-bottom palette in the leftmost column.
    The palette colors are distinct, non-grey singleton pixels, and the bottom
    row contains exactly one control pixel of each palette color. Grey pixels
    inside the grid are fixed blockers.

    Rule:
    Process colors in the palette's top-to-bottom order. From the matching
    bottom control pixel, repeatedly move upward through black cells and paint
    the visited cell with that color. If the cell above is non-black, including
    a grey blocker or an already drawn earlier path, move one cell right and
    paint it when that right cell is black; if the right cell is non-black or
    out of bounds, stop that path. Stop a path that reaches the top row.

    Key insight:
    The path order is supplied by the palette column, not by the controls'
    left-to-right order, and earlier paths become real obstacles for all later
    paths.
    """
    g = np.asarray(grid, dtype=int)
    palette, starts = _read_palette_and_controls(g)
    out, _events = _simulate_paths(g, palette, starts)
    return out


# --------------- generator ---------------


def _choose_core_columns(width: int) -> tuple[int, int]:
    choices: list[tuple[int, int]] = []
    for x_col in range(3, width - 5):
        for delta in range(4, 7):
            w_col = x_col + delta
            if w_col + 2 < width:
                choices.append((x_col, w_col))
    if not choices:
        raise ValueError("width is too small for the interaction template")
    return random.choice(choices)


def _place_palette_and_controls(
    grid: np.ndarray,
    palette: list[int],
    start_cols: dict[int, int],
) -> None:
    for r, color in enumerate(palette):
        grid[r, 0] = int(color)
    bottom = grid.shape[0] - 1
    for color, col in start_cols.items():
        grid[bottom, int(col)] = int(color)


def _protected_cells(
    height: int,
    width: int,
    palette_size: int,
    y_col: int,
    x_col: int,
    z_col: int,
    w_col: int,
    turn_y: int,
    turn_z: int,
) -> set[tuple[int, int]]:
    protected: set[tuple[int, int]] = set()

    for r in range(height):
        protected.add((r, 0))
    for r in range(palette_size + 1):
        for c in range(width):
            protected.add((r, c))
    for c in range(width):
        protected.add((height - 1, c))

    for r in range(0, height - 1):
        protected.add((r, x_col))
        protected.add((r, w_col))
        protected.add((r, w_col + 1))

    for r in range(turn_y + 1, height - 1):
        protected.add((r, y_col))
    for r in range(turn_z + 1, height - 1):
        protected.add((r, z_col))

    protected.add((turn_y, y_col))
    protected.add((turn_z, z_col))
    return protected


def _scatter_extra_blockers(
    grid: np.ndarray,
    protected: set[tuple[int, int]],
    count: int,
) -> None:
    height, width = map(int, grid.shape)
    candidates = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width)
        if grid[r, c] == BLACK and (r, c) not in protected
    ]
    random.shuffle(candidates)
    for r, c in candidates[:count]:
        grid[r, c] = GREY


def _independent_path_output(
    grid: np.ndarray,
    palette: list[int],
    starts: dict[int, tuple[int, int]],
) -> np.ndarray:
    out, _events = _simulate_paths(
        grid,
        palette,
        starts,
        use_current_state=False,
    )
    return out


def _semantically_valid(grid: np.ndarray) -> bool:
    try:
        palette, starts = _read_palette_and_controls(grid)
        expected, events = _simulate_paths(grid, palette, starts, record_events=True)
    except Exception:
        return False

    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(grid, expected):
        return False

    grey_deflects = [e for e in events if e["above_kind"] == "grey" and not e["terminates"]]
    previous_deflects = [
        e for e in events if e["above_kind"] == "previous_path" and not e["terminates"]
    ]
    previous_stops = [
        e
        for e in events
        if e["terminates"] and (e["above_kind"] == "previous_path" or e["right_kind"] == "previous_path")
    ]
    if not grey_deflects or not previous_deflects or not previous_stops:
        return False

    left_to_right = sorted(palette, key=lambda color: starts[color][1])
    if left_to_right == palette:
        return False

    try:
        by_controls, _ = _simulate_paths(grid, palette, starts, order=left_to_right)
        independent = _independent_path_output(grid, palette, starts)
        actual = main(grid)
    except Exception:
        return False

    return (
        np.array_equal(actual, expected)
        and not np.array_equal(by_controls, expected)
        and not np.array_equal(independent, expected)
    )


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(12, 20)
    width = random.randint(13, 24)
    x_col, w_col = _choose_core_columns(width)
    y_col = x_col - 1
    z_col = w_col - 1

    colors = random.sample(PATH_COLORS, 5)
    use_extra = bool(w_col + 3 < width and random.random() < 0.55)
    palette = [colors[0], colors[1], colors[2]]
    if use_extra:
        palette.append(colors[3])
    palette.append(colors[4])

    max_turn = height - 4
    min_turn = len(palette) + 1
    if min_turn > max_turn:
        return None
    turn_y = random.randint(min_turn, max_turn)
    turn_z = random.randint(min_turn, max_turn)

    grid = np.full((height, width), BLACK, dtype=int)
    start_cols = {
        colors[0]: x_col,
        colors[1]: z_col,
        colors[2]: w_col,
        colors[4]: y_col,
    }
    if use_extra:
        extra_candidates = [c for c in range(w_col + 3, width - 1)]
        if not extra_candidates:
            return None
        start_cols[colors[3]] = random.choice(extra_candidates)

    _place_palette_and_controls(grid, palette, start_cols)

    if grid[turn_y, y_col] != BLACK or grid[turn_z, z_col] != BLACK:
        return None
    grid[turn_y, y_col] = GREY
    grid[turn_z, z_col] = GREY

    protected = _protected_cells(
        height,
        width,
        len(palette),
        y_col,
        x_col,
        z_col,
        w_col,
        turn_y,
        turn_z,
    )
    max_extra = max(3, min(14, (height * width) // 18))
    extra_count = random.randint(3, max_extra)
    _scatter_extra_blockers(grid, protected, extra_count)

    if _semantically_valid(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid palette-ordered climbing path input")
