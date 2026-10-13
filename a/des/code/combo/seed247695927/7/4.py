import numpy as np


YELLOW = 4
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_RETRIES = 400
RNG = np.random.default_rng()


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bar_sets(grid: np.ndarray) -> tuple[set[frozenset[tuple[int, int]]], set[frozenset[tuple[int, int]]], int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    colors = [int(x) for x in np.unique(g) if int(x) != bg]
    assert len(colors) == 1, "expected exactly one foreground color"
    fg = colors[0]

    horizontal: set[frozenset[tuple[int, int]]] = set()
    vertical: set[frozenset[tuple[int, int]]] = set()
    for comp in _components_of_color(g, fg):
        assert len(comp) == 2, "foreground must be isolated two-pixel bars"
        (r1, c1), (r2, c2) = sorted(comp)
        if r1 == r2 and abs(c1 - c2) == 1:
            horizontal.add(frozenset(comp))
        elif c1 == c2 and abs(r1 - r2) == 1:
            vertical.add(frozenset(comp))
        else:
            raise AssertionError("each foreground component must be horizontal or vertical")
    return horizontal, vertical, bg, fg


def _has_bar(bar_set: set[frozenset[tuple[int, int]]], cells: tuple[tuple[int, int], tuple[int, int]]) -> bool:
    return frozenset(cells) in bar_set


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark each empty center surrounded by four separated two-pixel bars.

    A valid center is a background cell whose up/down/left/right neighbors are still
    background, and which has:
    - one horizontal 2-cell bar two rows above,
    - one horizontal 2-cell bar two rows below,
    - one vertical 2-cell bar two columns left,
    - one vertical 2-cell bar two columns right.

    Each bar may be offset by one cell along its own axis, so every bar cell lies at
    Manhattan distance 2 or 3 from the center.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    horizontal, vertical, bg, _fg = _bar_sets(g)
    h, w = g.shape

    for r in range(2, h - 2):
        for c in range(2, w - 2):
            if int(g[r, c]) != bg:
                continue
            if any(int(g[r + dr, c + dc]) != bg for dr, dc in DIRS4):
                continue

            top = _has_bar(horizontal, ((r - 2, c - 1), (r - 2, c))) or _has_bar(
                horizontal, ((r - 2, c), (r - 2, c + 1))
            )
            bottom = _has_bar(horizontal, ((r + 2, c - 1), (r + 2, c))) or _has_bar(
                horizontal, ((r + 2, c), (r + 2, c + 1))
            )
            left = _has_bar(vertical, ((r - 1, c - 2), (r, c - 2))) or _has_bar(
                vertical, ((r, c - 2), (r + 1, c - 2))
            )
            right = _has_bar(vertical, ((r - 1, c + 2), (r, c + 2))) or _has_bar(
                vertical, ((r, c + 2), (r + 1, c + 2))
            )

            if top and bottom and left and right:
                out[r, c] = YELLOW
    return out


def _can_place_bar(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != _get_bg(grid):
            return False
    for r, c in cells:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cell_set and int(grid[nr, nc]) != _get_bg(grid):
                return False
    return True


def _cross_bars(center: tuple[int, int], top_shift: int, bottom_shift: int, left_shift: int, right_shift: int) -> list[list[tuple[int, int]]]:
    r, c = center
    return [
        [(r - 2, c - 1 + top_shift), (r - 2, c + top_shift)],
        [(r + 2, c - 1 + bottom_shift), (r + 2, c + bottom_shift)],
        [(r - 1 + left_shift, c - 2), (r + left_shift, c - 2)],
        [(r - 1 + right_shift, c + 2), (r + right_shift, c + 2)],
    ]


def _paint_bar(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _candidate_extra_bar(rng: np.random.Generator, h: int, w: int) -> list[tuple[int, int]]:
    if rng.random() < 0.5:
        r = int(rng.integers(0, h))
        c = int(rng.integers(0, w - 1))
        return [(r, c), (r, c + 1)]
    r = int(rng.integers(0, h - 1))
    c = int(rng.integers(0, w))
    return [(r, c), (r + 1, c)]


def _expected_output(grid: np.ndarray, centers: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    for r, c in centers:
        out[r, c] = YELLOW
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(10, 23))
    w = int(rng.integers(10, 23))
    palette = [color for color in range(10) if color != YELLOW]
    bg = int(rng.choice(palette))
    fg_choices = [color for color in palette if color != bg]
    fg = int(rng.choice(fg_choices))
    grid = np.full((h, w), bg, dtype=int)

    max_centers = max(1, min(5, (h * w) // 55))
    cross_count = int(rng.integers(1, max_centers + 1))
    centers: list[tuple[int, int]] = []

    for _ in range(cross_count):
        placed = False
        for _ in range(120):
            center = (int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2)))
            shifts = [int(rng.integers(0, 2)) for _ in range(4)]
            bars = _cross_bars(center, *shifts)
            if any(not _can_place_bar(grid, bar) for bar in bars):
                continue
            if int(grid[center]) != bg:
                continue
            if any(
                not (0 <= center[0] + dr < h and 0 <= center[1] + dc < w) or int(grid[center[0] + dr, center[1] + dc]) != bg
                for dr, dc in DIRS4
            ):
                continue
            for bar in bars:
                _paint_bar(grid, bar, fg)
            centers.append(center)
            placed = True
            break
        if not placed:
            return None

    target_extra = int(rng.integers(0, cross_count * 3 + 4))
    placed_extra = 0
    for _ in range(target_extra * 12 + 20):
        if placed_extra >= target_extra:
            break
        bar = _candidate_extra_bar(rng, h, w)
        if _can_place_bar(grid, bar):
            _paint_bar(grid, bar, fg)
            placed_extra += 1

    if np.count_nonzero(grid == fg) >= (h * w) // 2:
        return None

    observed = main(grid)
    expected = _expected_output(grid, centers)
    if not np.array_equal(observed, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
