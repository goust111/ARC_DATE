import numpy as np


BLACK = 0
COLOR_POOL = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
NETWORK_H_RANGE = (9, 18)
NETWORK_W_RANGE = (9, 18)
LINE_COUNT_CAP = 4
MAX_GENERATION_ATTEMPTS = 300

_RNG = np.random.default_rng()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve path crossings using the top-row legend order and drop the legend row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2

    legend = [int(color) for color in g[0] if int(color) != BLACK]
    rank = {color: idx for idx, color in enumerate(legend)}
    fallback_rank = len(rank)

    network = g[1:].copy()
    out = network.copy()
    h, w = network.shape

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(network[r, c]) == BLACK:
                continue

            vertical = None
            up = int(network[r - 1, c])
            down = int(network[r + 1, c])
            if up != BLACK and up == down:
                vertical = up

            horizontal = None
            left = int(network[r, c - 1])
            right = int(network[r, c + 1])
            if left != BLACK and left == right:
                horizontal = left

            if vertical is None or horizontal is None or vertical == horizontal:
                continue

            if rank.get(vertical, fallback_rank) < rank.get(horizontal, fallback_rank):
                out[r, c] = vertical
            else:
                out[r, c] = horizontal

    return out


def _randint(bounds: tuple[int, int]) -> int:
    return int(_RNG.integers(bounds[0], bounds[1] + 1))


def _sample_spaced_positions(length: int, count: int) -> np.ndarray:
    interior = length - 2
    max_count = (interior + 1) // 2
    assert 1 <= count <= max_count

    compressed = np.sort(_RNG.choice(interior - count + 1, size=count, replace=False))
    return 1 + compressed + np.arange(count, dtype=int)


def _sample_line_count(length: int) -> int:
    interior = length - 2
    max_count = min(LINE_COUNT_CAP, (interior + 1) // 2)
    return int(_RNG.integers(1, max_count + 1))


def _crossing_visibility(expected: list[int], alternate: list[int], mode: int) -> list[int]:
    if mode == 0:
        return expected[:]
    if mode == 1:
        return alternate[:]

    visible = [
        expected[i] if int(_RNG.integers(0, 2)) == 0 else alternate[i]
        for i in range(len(expected))
    ]
    if len(visible) >= 2:
        same_as_expected = [visible[i] == expected[i] for i in range(len(visible))]
        if all(same_as_expected):
            visible[0] = alternate[0]
        elif not any(same_as_expected):
            visible[0] = expected[0]
    return visible


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    net_h = _randint(NETWORK_H_RANGE)
    net_w = _randint(NETWORK_W_RANGE)

    horizontal_count = _sample_line_count(net_h)
    vertical_count = _sample_line_count(net_w)
    total_paths = horizontal_count + vertical_count

    colors = _RNG.choice(COLOR_POOL, size=total_paths, replace=False).astype(int)
    horizontal_colors = colors[:horizontal_count]
    vertical_colors = colors[horizontal_count:]

    horizontal_rows = _sample_spaced_positions(net_h, horizontal_count)
    vertical_cols = _sample_spaced_positions(net_w, vertical_count)

    legend_colors = _RNG.permutation(colors).astype(int)
    rank = {int(color): idx for idx, color in enumerate(legend_colors)}

    row_to_color = {int(r): int(color) for r, color in zip(horizontal_rows, horizontal_colors)}
    col_to_color = {int(c): int(color) for c, color in zip(vertical_cols, vertical_colors)}

    input_network = np.zeros((net_h, net_w), dtype=int)
    output_network = np.zeros((net_h, net_w), dtype=int)

    crossing_cells: list[tuple[int, int]] = []
    expected_cross_colors: list[int] = []
    alternate_cross_colors: list[int] = []

    for r in range(net_h):
        row_color = row_to_color.get(r)
        for c in range(net_w):
            col_color = col_to_color.get(c)
            if row_color is not None and col_color is not None:
                crossing_cells.append((r, c))
                if rank[row_color] < rank[col_color]:
                    expected = row_color
                    alternate = col_color
                else:
                    expected = col_color
                    alternate = row_color
                expected_cross_colors.append(expected)
                alternate_cross_colors.append(alternate)
                output_network[r, c] = expected
            elif row_color is not None:
                input_network[r, c] = row_color
                output_network[r, c] = row_color
            elif col_color is not None:
                input_network[r, c] = col_color
                output_network[r, c] = col_color

    if not crossing_cells:
        return None

    mode = int(_RNG.integers(0, 3))
    visible_cross_colors = _crossing_visibility(expected_cross_colors, alternate_cross_colors, mode)

    for (r, c), visible_color in zip(crossing_cells, visible_cross_colors):
        input_network[r, c] = visible_color

    legend_row = np.zeros(net_w, dtype=int)
    legend_positions = np.sort(_RNG.choice(net_w, size=total_paths, replace=False))
    for pos, color in zip(legend_positions, legend_colors):
        legend_row[int(pos)] = int(color)

    inp = np.vstack([legend_row, input_network])
    out = output_network

    if not np.array_equal(main(inp), out):
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input")
