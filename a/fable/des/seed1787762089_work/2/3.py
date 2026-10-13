import numpy as np


_RNG = np.random.default_rng()

_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
_NM_CHOICES = tuple(
    (n, m) for n in range(3, 6) for m in range(2, 5) if n != m
)
_HEIGHT_RANGE = (12, 18)
_WIDTH_RANGE = (14, 20)
_MAX_GENERATION_ATTEMPTS = 300


def _border_coords(height: int, width: int) -> tuple[tuple[int, int], ...]:
    coords = [(0, c) for c in range(width)]
    coords += [(r, width - 1) for r in range(1, height)]
    coords += [(height - 1, c) for c in range(width - 2, -1, -1)]
    coords += [(r, 0) for r in range(height - 2, 0, -1)]
    return tuple(coords)


def _border_side_ids(height: int, width: int) -> tuple[int, ...]:
    return (
        (0,) * width
        + (1,) * (height - 1)
        + (2,) * (width - 1)
        + (3,) * (height - 2)
    )


def _count_components(mask: np.ndarray) -> int:
    seen = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    count = 0

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            count += 1
            seen[row, col] = True
            stack = [(row, col)]
            while stack:
                cur_row, cur_col = stack.pop()
                for drow in (-1, 0, 1):
                    for dcol in (-1, 0, 1):
                        if drow == 0 and dcol == 0:
                            continue
                        nxt_row = cur_row + drow
                        nxt_col = cur_col + dcol
                        if not (0 <= nxt_row < height and 0 <= nxt_col < width):
                            continue
                        if not mask[nxt_row, nxt_col] or seen[nxt_row, nxt_col]:
                            continue
                        seen[nxt_row, nxt_col] = True
                        stack.append((nxt_row, nxt_col))
    return count


def _infer_color(grid: np.ndarray) -> int:
    colors = [int(color) for color in np.unique(grid) if int(color) != 0]
    assert len(colors) == 1, "expected exactly one non-black color"
    return colors[0]


def _find_segment_start(border_values: np.ndarray, color: int) -> tuple[int, int]:
    mask = border_values == color
    length = int(np.count_nonzero(mask))
    assert length >= 1, "expected a colored border segment"

    starts = [
        idx
        for idx in range(mask.size)
        if bool(mask[idx]) and not bool(mask[(idx - 1) % mask.size])
    ]
    assert len(starts) == 1, "expected exactly one border segment"
    return starts[0], length


def _paint_periodic_border(
    out: np.ndarray, color: int, start: int, segment_len: int, gap_len: int
) -> None:
    border = _border_coords(*out.shape)
    period = segment_len + gap_len
    assert len(border) % period == 0, "border perimeter must match the periodic step"

    for row, col in border:
        out[row, col] = 0
    for base in range(0, len(border), period):
        for offset in range(segment_len):
            row, col = border[(start + base + offset) % len(border)]
            out[row, col] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Repeat the border segment with gaps equal to the count of interior same-color objects."""
    g = np.asarray(grid, dtype=int)
    color = _infer_color(g)
    border = _border_coords(*g.shape)
    border_values = np.array([g[row, col] for row, col in border], dtype=int)

    start, segment_len = _find_segment_start(border_values, color)
    inner_mask = g[1:-1, 1:-1] == color
    gap_len = _count_components(inner_mask)
    assert gap_len >= 1, "expected at least one interior object"

    out = g.copy()
    out[out == color] = 0
    _paint_periodic_border(out, color, start, segment_len, gap_len)
    return out


def _valid_dimensions(period: int) -> tuple[tuple[int, int], ...]:
    dims = []
    for height in range(_HEIGHT_RANGE[0], _HEIGHT_RANGE[1] + 1):
        for width in range(_WIDTH_RANGE[0], _WIDTH_RANGE[1] + 1):
            if (2 * (height + width) - 4) % period == 0:
                dims.append((height, width))
    return tuple(dims)


_DIMS_BY_PERIOD = {period: _valid_dimensions(period) for period in range(5, 10)}


def _segment_crosses_corner(height: int, width: int, start: int, segment_len: int) -> bool:
    side_ids = _border_side_ids(height, width)
    used_sides = {
        side_ids[(start + offset) % len(side_ids)] for offset in range(segment_len)
    }
    return len(used_sides) > 1


def _choose_segment_start(height: int, width: int, segment_len: int) -> int:
    total = 2 * (height + width) - 4
    want_corner_cross = bool(_RNG.integers(0, 2))
    candidates = [
        start
        for start in range(total)
        if _segment_crosses_corner(height, width, start, segment_len) == want_corner_cross
    ]
    if not candidates:
        candidates = list(range(total))
    return int(candidates[int(_RNG.integers(0, len(candidates)))])


def _choose_singletons(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    cells = [(row, col) for row in range(2, height - 2) for col in range(2, width - 2)]
    order = _RNG.permutation(len(cells))
    chosen: list[tuple[int, int]] = []

    for idx in order:
        row, col = cells[int(idx)]
        if any(max(abs(row - prev_row), abs(col - prev_col)) <= 1 for prev_row, prev_col in chosen):
            continue
        chosen.append((row, col))
        if len(chosen) == count:
            return chosen
    return None


def _gen_attempt() -> np.ndarray | None:
    segment_len, gap_len = _NM_CHOICES[int(_RNG.integers(0, len(_NM_CHOICES)))]
    period = segment_len + gap_len
    dims = _DIMS_BY_PERIOD[period]
    if not dims:
        return None

    height, width = dims[int(_RNG.integers(0, len(dims)))]
    color = int(_COLOR_POOL[int(_RNG.integers(0, len(_COLOR_POOL)))])
    start = _choose_segment_start(height, width, segment_len)
    border = _border_coords(height, width)

    grid = np.zeros((height, width), dtype=int)
    for offset in range(segment_len):
        row, col = border[(start + offset) % len(border)]
        grid[row, col] = color

    points = _choose_singletons(height, width, gap_len)
    if points is None:
        return None
    for row, col in points:
        grid[row, col] = color

    border_values = np.array([grid[row, col] for row, col in border], dtype=int)
    found_start, found_len = _find_segment_start(border_values, color)
    assert found_start == start and found_len == segment_len
    assert _count_components(grid[1:-1, 1:-1] == color) == gap_len

    out = main(grid)
    assert out.shape == grid.shape
    assert np.count_nonzero(out[1:-1, 1:-1] == color) == 0
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
