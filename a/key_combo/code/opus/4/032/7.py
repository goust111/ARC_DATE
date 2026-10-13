import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    ORANGE = 7


GRID_SIZES = (9, 11, 13)
COMPONENT_COUNTS = (2, 3, 4, 5, 6)
MAX_RETRIES = 500


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _green_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(grid, dtype=int) == Color.GREEN
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(sorted(comp))
    return components


def _build_output(shape: tuple[int, int], component_count: int) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), Color.BLACK, dtype=int)
    out[::2, :] = Color.BLUE
    out[:, ::2] = Color.BLUE

    cell_rows = list(range(1, h, 2))
    cell_cols = list(range(1, w, 2))
    last_row = len(cell_rows) - 1
    last_col = len(cell_cols) - 1

    for row_i, r in enumerate(cell_rows):
        for col_i, c in enumerate(cell_cols):
            if row_i in (0, last_row) and col_i in (0, last_col):
                out[r, c] = Color.RED if component_count % 2 == 1 else Color.YELLOW
            elif component_count % 2 == 0 and 0 < row_i < last_row and 0 < col_i < last_col:
                out[r, c] = Color.ORANGE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count green 8-connected components before overlaying periodic blue grid lines, then fill the resulting odd-odd cells from that count parity.

    Input structure:
    - The input is a square odd-sized grid on a black background with scattered green pixels.
    - Only the green pixels matter as objects; they form several separate 8-connected components and may lie on either future grid-line positions or future cell positions.

    Rule:
    - Count the number of 8-connected green components.
    - Erase the green pixels, draw blue grid lines on every even-indexed row and every even-indexed column, and treat the remaining odd-row/odd-column pixels as table cells.
    - If the component count is odd, color only the four corner cells red.
    - If the component count is even, color the four corner cells yellow and every strictly interior cell orange.
    - Blue grid-line pixels always remain blue and all other cells remain black.

    Key insight:
    - The visible green pattern is only a temporary code: its 8-connected component count determines which table-filling mode to apply after the green pixels disappear under the blue grid scaffold.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == w and h % 2 == 1, "expected an odd-sized square grid"
    assert max(h, w) <= 30
    assert set(np.unique(g)).issubset({Color.BLACK, Color.GREEN}), "expected only black and green in the input"

    component_count = len(_green_components(g))
    assert 2 <= component_count <= 6, "expected 2 to 6 green components"
    return _build_output(g.shape, component_count)


def _normalize_shape(coords: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _transform_shape(coords: tuple[tuple[int, int], ...], code: int) -> tuple[tuple[int, int], ...]:
    pts = list(coords)
    transformed = []
    for r, c in pts:
        if code == 0:
            rr, cc = r, c
        elif code == 1:
            rr, cc = r, -c
        elif code == 2:
            rr, cc = -r, c
        elif code == 3:
            rr, cc = -r, -c
        elif code == 4:
            rr, cc = c, r
        elif code == 5:
            rr, cc = c, -r
        elif code == 6:
            rr, cc = -c, r
        else:
            rr, cc = -c, -r
        transformed.append((rr, cc))
    return _normalize_shape(set(transformed))


def _random_shape(size: int, rng: np.random.Generator) -> tuple[tuple[int, int], ...]:
    coords = {(0, 0)}
    while len(coords) < size:
        frontier = set()
        for r, c in coords:
            for nr, nc in _neighbors8(r, c):
                if (nr, nc) not in coords:
                    frontier.add((nr, nc))
        frontier_list = list(frontier)
        coords.add(frontier_list[int(rng.integers(0, len(frontier_list)))])
    shape = _normalize_shape(coords)
    return _transform_shape(shape, int(rng.integers(0, 8)))


def _choose_sizes(component_count: int, rng: np.random.Generator) -> list[int]:
    max_total = min(10, 4 * component_count)
    totals = [
        total
        for total in range(component_count + 1, max_total + 1)
        if total % 2 != component_count % 2
    ]
    assert totals
    for _ in range(200):
        total = int(rng.choice(totals))
        extras = total - component_count
        bump = [0] * component_count
        for _ in range(extras):
            options = [i for i, extra in enumerate(bump) if extra < 3]
            if not options:
                break
            idx = int(rng.choice(options))
            bump[idx] += 1
        sizes = [1 + extra for extra in bump]
        if (
            sum(sizes) == total
            and any(size == 1 for size in sizes)
            and any(size > 1 for size in sizes)
        ):
            rng.shuffle(sizes)
            return sizes
    raise RuntimeError("failed to sample component sizes")


def _candidate_positions(size: int, shape: tuple[tuple[int, int], ...]):
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    for base_r in range(size - max_r):
        for base_c in range(size - max_c):
            yield [(base_r + r, base_c + c) for r, c in shape]


def _place_components(
    size: int,
    shapes: list[tuple[tuple[int, int], ...]],
    rng: np.random.Generator,
) -> np.ndarray | None:
    grid = np.full((size, size), Color.BLACK, dtype=int)
    occupied: list[tuple[int, int]] = []

    for shape in sorted(shapes, key=len, reverse=True):
        candidates = list(_candidate_positions(size, shape))
        rng.shuffle(candidates)
        placed = None
        for pts in candidates:
            if any((r, c) in ((0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)) for r, c in pts):
                continue
            valid = True
            for r, c in pts:
                for qr, qc in occupied:
                    if max(abs(r - qr), abs(c - qc)) <= 1:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                placed = pts
                break
        if placed is None:
            return None
        for r, c in placed:
            grid[r, c] = Color.GREEN
            occupied.append((r, c))
    return grid


def _semantic_checks(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.shape[0] == g.shape[1]
    assert g.shape[0] in GRID_SIZES
    assert set(np.unique(g)).issubset({Color.BLACK, Color.GREEN})

    green_pixels = [tuple(x) for x in np.argwhere(g == Color.GREEN)]
    assert 3 <= len(green_pixels) <= 10
    corners = {(0, 0), (0, g.shape[1] - 1), (g.shape[0] - 1, 0), (g.shape[0] - 1, g.shape[1] - 1)}
    assert not any(pt in corners for pt in green_pixels)

    components = _green_components(g)
    count = len(components)
    assert 2 <= count <= 6
    sizes = sorted(len(comp) for comp in components)
    assert all(1 <= size <= 4 for size in sizes)
    assert any(size == 1 for size in sizes)
    assert any(size > 1 for size in sizes)
    assert len(green_pixels) % 2 != count % 2

    has_cell_pixel = any(r % 2 == 1 and c % 2 == 1 for r, c in green_pixels)
    has_line_pixel = any(r % 2 == 0 or c % 2 == 0 for r, c in green_pixels)
    assert has_cell_pixel and has_line_pixel

    out = main(g)
    expected = _build_output(g.shape, count)
    assert np.array_equal(out, expected)

    cell_count = (g.shape[0] - 1) // 2
    if count % 2 == 1:
        assert int(np.sum(out == Color.RED)) == 4
        assert int(np.sum(out == Color.YELLOW)) == 0
        assert int(np.sum(out == Color.ORANGE)) == 0
    else:
        assert int(np.sum(out == Color.YELLOW)) == 4
        assert int(np.sum(out == Color.ORANGE)) == max(cell_count - 2, 0) ** 2
        assert int(np.sum(out == Color.RED)) == 0


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    size = int(rng.choice(GRID_SIZES))
    parity = int(rng.integers(0, 2))
    eligible_counts = [count for count in COMPONENT_COUNTS if count % 2 == parity]
    component_count = int(rng.choice(eligible_counts))
    sizes = _choose_sizes(component_count, rng)
    shapes = [_random_shape(size_i, rng) for size_i in sizes]
    grid = _place_components(size, shapes, rng)
    if grid is None:
        return None
    try:
        _semantic_checks(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
