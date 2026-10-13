import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MIN_SCENE_SIZE = 9
MAX_SCENE_SIZE = 20
MAX_GENERATION_ATTEMPTS = 300

VERTICAL_SIDES = {"top", "bottom"}
HORIZONTAL_SIDES = {"left", "right"}


def _port_to_coord(port: tuple[str, int], h: int, w: int) -> tuple[int, int]:
    side, index = port
    if side == "top":
        return 0, index
    if side == "bottom":
        return h - 1, index
    if side == "left":
        return index, 0
    if side == "right":
        return index, w - 1
    raise AssertionError(f"unknown side: {side}")


def _coord_to_port(r: int, c: int, h: int, w: int) -> tuple[str, int]:
    if r == 0 and 0 < c < w - 1:
        return "top", c
    if r == h - 1 and 0 < c < w - 1:
        return "bottom", c
    if c == 0 and 0 < r < h - 1:
        return "left", r
    if c == w - 1 and 0 < r < h - 1:
        return "right", r
    raise AssertionError("ports must lie on a non-corner border cell")


def _trace_path(h: int, w: int, port_a: tuple[str, int], port_b: tuple[str, int]) -> set[tuple[int, int]]:
    side_a, idx_a = port_a
    side_b, idx_b = port_b

    if {side_a, side_b} == {"top", "bottom"}:
        assert idx_a == idx_b, "opposite vertical ports must share a column"
        return {(r, idx_a) for r in range(h)}

    if {side_a, side_b} == {"left", "right"}:
        assert idx_a == idx_b, "opposite horizontal ports must share a row"
        return {(idx_a, c) for c in range(w)}

    assert (side_a in VERTICAL_SIDES and side_b in HORIZONTAL_SIDES) or (
        side_b in VERTICAL_SIDES and side_a in HORIZONTAL_SIDES
    ), "each pair must be adjacent-edge or aligned opposite-edge ports"

    if side_a in VERTICAL_SIDES:
        vert_side, vert_index = side_a, idx_a
        hori_side, hori_index = side_b, idx_b
    else:
        vert_side, vert_index = side_b, idx_b
        hori_side, hori_index = side_a, idx_a

    row = hori_index
    col = vert_index
    cells: set[tuple[int, int]] = set()

    if vert_side == "top":
        for rr in range(0, row + 1):
            cells.add((rr, col))
    else:
        for rr in range(row, h):
            cells.add((rr, col))

    if hori_side == "left":
        for cc in range(0, col + 1):
            cells.add((row, cc))
    else:
        for cc in range(col, w):
            cells.add((row, cc))

    return cells


def _paths_from_grid(grid: np.ndarray) -> dict[int, set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 3 and w >= 3
    assert np.all(g[1:-1, 1:-1] == BG), "the interior must be empty in the input"

    ports_by_color: dict[int, list[tuple[str, int]]] = {}
    for rr, cc in np.argwhere(g != BG):
        port = _coord_to_port(int(rr), int(cc), h, w)
        color = int(g[rr, cc])
        ports_by_color.setdefault(color, []).append(port)

    assert ports_by_color, "expected at least one color pair"

    paths: dict[int, set[tuple[int, int]]] = {}
    for color, ports in ports_by_color.items():
        assert len(ports) == 2, "each port color must appear exactly twice"
        assert ports[0][0] != ports[1][0], "paired ports must lie on different edges"
        paths[color] = _trace_path(h, w, ports[0], ports[1])

    return paths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color border-port pair with its implied orthogonal path and recolor overlaps by the crossing rule.

    Input structure:
    - The background is black.
    - Non-black cells are one-pixel ports placed on the border, never in the corners.
    - Each non-black color appears exactly twice on two different edges, and the interior is otherwise empty.

    Rule:
    - For each same-color pair, draw the unique width-1 path implied by the edges: aligned opposite-edge ports form a straight line, while adjacent-edge ports form a one-turn orthogonal path whose vertical leg follows the vertical-edge port and whose horizontal leg follows the horizontal-edge port.
    - Draw every path on a black output grid using its own color.
    - If exactly two path colors share a cell, paint that cell with the unique remaining port color when one exists; otherwise paint it grey.
    - If three or more paths share a cell, paint it grey.

    Key insight:
    - The ports determine all hidden paths uniquely; the real reasoning step is to aggregate all paths first and then color shared cells by the complementary-color rule instead of by overwrite order.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    paths = _paths_from_grid(g)
    all_colors = set(paths)

    cell_to_colors: dict[tuple[int, int], set[int]] = {}
    for color, cells in paths.items():
        for cell in cells:
            cell_to_colors.setdefault(cell, set()).add(color)

    out = np.full((h, w), BG, dtype=int)
    for (rr, cc), colors in cell_to_colors.items():
        if len(colors) == 1:
            out[rr, cc] = next(iter(colors))
            continue
        if len(colors) >= 3:
            out[rr, cc] = GREY
            continue

        remaining = [color for color in all_colors if color not in colors]
        out[rr, cc] = remaining[0] if len(remaining) == 1 else GREY

    return out


def _render_input(h: int, w: int, pairs: list[tuple[int, tuple[str, int], tuple[str, int]]]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    used_coords: set[tuple[int, int]] = set()

    for color, port_a, port_b in pairs:
        for port in (port_a, port_b):
            rr, cc = _port_to_coord(port, h, w)
            assert (rr, cc) not in used_coords, "ports must occupy distinct cells"
            used_coords.add((rr, cc))
            grid[rr, cc] = color

    return grid


def _choose_colors(rng: np.random.Generator, count: int) -> list[int]:
    return [int(color) for color in rng.choice(np.array(PALETTE, dtype=int), size=count, replace=False)]


def _sample_size(rng: np.random.Generator, min_h: int = MIN_SCENE_SIZE, min_w: int = MIN_SCENE_SIZE) -> tuple[int, int]:
    h = int(rng.integers(min_h, MAX_SCENE_SIZE + 1))
    w = int(rng.integers(min_w, MAX_SCENE_SIZE + 1))
    return h, w


def _scene_l_grey(rng: np.random.Generator) -> np.ndarray:
    h, w = _sample_size(rng)
    rows = sorted(int(v) for v in rng.choice(np.arange(2, h - 1), size=2, replace=False))
    cols = sorted(int(v) for v in rng.choice(np.arange(2, w - 1), size=2, replace=False))
    colors = _choose_colors(rng, 2)

    r_low, r_high = rows
    c_left, c_right = cols
    pairs = [
        (colors[0], ("top", c_left), ("left", r_high)),
        (colors[1], ("bottom", c_right), ("left", r_low)),
    ]
    return _render_input(h, w, pairs)


def _scene_third_color(rng: np.random.Generator) -> np.ndarray:
    h, w = _sample_size(rng)
    rows = sorted(int(v) for v in rng.choice(np.arange(2, h - 1), size=2, replace=False))
    cols = sorted(int(v) for v in rng.choice(np.arange(2, w - 1), size=3, replace=False))
    colors = _choose_colors(rng, 3)

    r_low, r_high = rows
    c_left, c_mid, c_far = cols
    pairs = [
        (colors[0], ("top", c_left), ("left", r_high)),
        (colors[1], ("bottom", c_mid), ("left", r_low)),
        (colors[2], ("top", c_far), ("bottom", c_far)),
    ]
    return _render_input(h, w, pairs)


def _scene_triple_grey(rng: np.random.Generator) -> np.ndarray:
    h, w = _sample_size(rng)
    r_cross = int(rng.integers(3, h - 2))
    top_cols = sorted(int(v) for v in rng.choice(np.arange(2, w - 1), size=2, replace=False))
    colors = _choose_colors(rng, 3)

    c_shared, c_far = top_cols
    pairs = [
        (colors[0], ("top", c_shared), ("right", r_cross + 1)),
        (colors[1], ("bottom", c_shared), ("left", r_cross - 1)),
        (colors[2], ("top", c_far), ("left", r_cross)),
    ]
    return _render_input(h, w, pairs)


def _scene_straight_grey(rng: np.random.Generator) -> np.ndarray:
    h, w = _sample_size(rng)
    row = int(rng.integers(2, h - 1))
    col = int(rng.integers(2, w - 1))
    colors = _choose_colors(rng, 2)

    pairs = [
        (colors[0], ("top", col), ("bottom", col)),
        (colors[1], ("left", row), ("right", row)),
    ]
    return _render_input(h, w, pairs)


SCENE_BUILDERS = (
    _scene_l_grey,
    _scene_third_color,
    _scene_triple_grey,
    _scene_straight_grey,
)


def _apply_random_symmetry(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.rot90(grid, int(rng.integers(0, 4)))
    if bool(int(rng.integers(0, 2))):
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


def _validate_generated_grid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30
    paths = _paths_from_grid(grid)
    assert 2 <= len(paths) <= 3
    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out != BG)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    try:
        builder = SCENE_BUILDERS[int(rng.integers(0, len(SCENE_BUILDERS)))]
        grid = builder(rng)
        grid = _apply_random_symmetry(grid, rng)
        _validate_generated_grid(grid)
        return grid
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
