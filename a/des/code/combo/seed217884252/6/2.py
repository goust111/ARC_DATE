import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

LEGEND_SEQUENCE = (BLUE, GREEN, RED, YELLOW)
RECTANGLE_COLORS = (BLACK, GRAY, MAGENTA, CYAN, MAROON)

MIN_GRID_H = 12
MAX_GRID_H = 18
MIN_PLAY_W = 10
MAX_PLAY_W = 16
LEGEND_GAP = 1
MIN_TOP_SIZE = 2
MAX_TOP_SIZE = 5
MAX_RECTANGLES = len(RECTANGLE_COLORS)
MAX_GENERATION_ATTEMPTS = 300
GEOMETRY_RETRIES = 120
EXTRA_RECT_PROB = 0.5
MARGIN_MIN = 1
MARGIN_MAX = 2


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_legend_mask(grid: np.ndarray, bg: int) -> np.ndarray:
    h, w = grid.shape
    matches: list[tuple[int, int]] = []
    seq_len = len(LEGEND_SEQUENCE)
    for col in (0, w - 1):
        for r0 in range(h - seq_len + 1):
            if tuple(int(x) for x in grid[r0 : r0 + seq_len, col]) != LEGEND_SEQUENCE:
                continue
            before_ok = r0 == 0 or int(grid[r0 - 1, col]) == bg
            after_ok = r0 + seq_len == h or int(grid[r0 + seq_len, col]) == bg
            if before_ok and after_ok:
                matches.append((r0, col))
    assert len(matches) == 1, "expected exactly one edge legend"
    r0, col = matches[0]
    mask = np.zeros_like(grid, dtype=bool)
    mask[r0 : r0 + len(LEGEND_SEQUENCE), col] = True
    return mask


def _rectangles_from_visible(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    rectangles: list[tuple[int, int, int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        assert coords.size > 0
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        rectangles.append((int(r0), int(c0), int(r1), int(c1)))
    return rectangles


def _coverage_map(shape: tuple[int, int], rectangles: list[tuple[int, int, int, int]]) -> np.ndarray:
    cover = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in rectangles:
        cover[r0 : r1 + 1, c0 : c1 + 1] += 1
    return cover


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each cell by how many inferred rectangles cover it, then erase the legend."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = ORANGE
    legend_mask = _find_legend_mask(g, bg)

    visible = g.copy()
    visible[legend_mask] = bg
    rectangles = _rectangles_from_visible(visible, bg)

    cover = _coverage_map(g.shape, rectangles)
    out = np.full(g.shape, bg, dtype=int)
    out[cover == 1] = BLUE
    out[cover == 2] = GREEN
    out[cover == 3] = RED
    out[cover >= 4] = YELLOW
    return out


def _paint_rectangles(
    shape: tuple[int, int], rectangles: list[tuple[int, int, int, int]], colors: list[int]
) -> np.ndarray:
    grid = np.full(shape, ORANGE, dtype=int)
    for (r0, c0, r1, c1), color in zip(rectangles, colors):
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    return grid


def _paint_legend(grid: np.ndarray, side: str, start_row: int) -> None:
    col = 0 if side == "left" else grid.shape[1] - 1
    for offset, color in enumerate(LEGEND_SEQUENCE):
        grid[start_row + offset, col] = color


def _visible_bbox_matches(
    rendered: np.ndarray, rectangle: tuple[int, int, int, int], color: int
) -> bool:
    coords = np.argwhere(rendered == color)
    if coords.size == 0:
        return False
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return (int(r0), int(c0), int(r1), int(c1)) == rectangle


def _boxes_touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _nested_core_rectangles(
    rng: np.random.Generator, grid_h: int, play_c0: int, play_c1: int, count: int
) -> list[tuple[int, int, int, int]] | None:
    play_w = play_c1 - play_c0 + 1
    for _ in range(GEOMETRY_RETRIES):
        margins: list[tuple[int, int, int, int]] = []
        total_h = 0
        total_w = 0
        for _ in range(count - 1):
            top = int(rng.integers(MARGIN_MIN, MARGIN_MAX + 1))
            bottom = int(rng.integers(MARGIN_MIN, MARGIN_MAX + 1))
            left = int(rng.integers(MARGIN_MIN, MARGIN_MAX + 1))
            right = int(rng.integers(MARGIN_MIN, MARGIN_MAX + 1))
            margins.append((top, bottom, left, right))
            total_h += top + bottom
            total_w += left + right

        max_top_h = min(MAX_TOP_SIZE, grid_h - total_h)
        max_top_w = min(MAX_TOP_SIZE, play_w - total_w)
        if max_top_h < MIN_TOP_SIZE or max_top_w < MIN_TOP_SIZE:
            continue

        top_h = int(rng.integers(MIN_TOP_SIZE, max_top_h + 1))
        top_w = int(rng.integers(MIN_TOP_SIZE, max_top_w + 1))
        outer_h = top_h + total_h
        outer_w = top_w + total_w
        if outer_h > grid_h or outer_w > play_w:
            continue

        outer_r0 = int(rng.integers(0, grid_h - outer_h + 1))
        outer_c0 = int(rng.integers(play_c0, play_c1 - outer_w + 2))
        outer_r1 = outer_r0 + outer_h - 1
        outer_c1 = outer_c0 + outer_w - 1

        rectangles = [(outer_r0, outer_c0, outer_r1, outer_c1)]
        cur_r0, cur_c0, cur_r1, cur_c1 = outer_r0, outer_c0, outer_r1, outer_c1
        valid = True
        for top, bottom, left, right in margins:
            cur_r0 += top
            cur_r1 -= bottom
            cur_c0 += left
            cur_c1 -= right
            if cur_r0 > cur_r1 or cur_c0 > cur_c1:
                valid = False
                break
            rectangles.append((cur_r0, cur_c0, cur_r1, cur_c1))
        if valid and len(rectangles) == count:
            return rectangles
    return None


def _sample_extra_rectangle(
    rng: np.random.Generator,
    grid_h: int,
    play_c0: int,
    play_c1: int,
    occupied: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    play_w = play_c1 - play_c0 + 1
    for _ in range(GEOMETRY_RETRIES):
        height = int(rng.integers(2, min(6, grid_h) + 1))
        width = int(rng.integers(2, min(6, play_w) + 1))
        r0 = int(rng.integers(0, grid_h - height + 1))
        c0 = int(rng.integers(play_c0, play_c1 - width + 2))
        rect = (r0, c0, r0 + height - 1, c0 + width - 1)
        if all(not _boxes_touch(rect, other, pad=1) for other in occupied):
            return rect
    return None


def _expected_output(shape: tuple[int, int], rectangles: list[tuple[int, int, int, int]]) -> np.ndarray:
    cover = _coverage_map(shape, rectangles)
    out = np.full(shape, ORANGE, dtype=int)
    out[cover == 1] = BLUE
    out[cover == 2] = GREEN
    out[cover == 3] = RED
    out[cover >= 4] = YELLOW
    return out


def _build_sample(rng: np.random.Generator) -> np.ndarray | None:
    grid_h = int(rng.integers(MIN_GRID_H, MAX_GRID_H + 1))
    play_w = int(rng.integers(MIN_PLAY_W, MAX_PLAY_W + 1))
    total_w = play_w + LEGEND_GAP + 1
    legend_side = "left" if rng.random() < 0.5 else "right"
    play_c0 = 2 if legend_side == "left" else 0
    play_c1 = total_w - 1 if legend_side == "left" else total_w - 3

    branch = int(rng.integers(2, 5))
    core_count = branch if branch < 4 else int(rng.choice([4, 5]))
    rectangles = _nested_core_rectangles(rng, grid_h, play_c0, play_c1, core_count)
    if rectangles is None:
        return None

    if core_count < MAX_RECTANGLES and rng.random() < EXTRA_RECT_PROB:
        extra = _sample_extra_rectangle(rng, grid_h, play_c0, play_c1, rectangles)
        if extra is not None:
            rectangles = rectangles + [extra]

    colors = list(rng.choice(np.array(RECTANGLE_COLORS), size=len(rectangles), replace=False))
    grid = _paint_rectangles((grid_h, total_w), rectangles, colors)

    legend_row = int(rng.integers(0, grid_h - len(LEGEND_SEQUENCE) + 1))
    _paint_legend(grid, legend_side, legend_row)

    for rectangle, color in zip(rectangles, colors):
        if not _visible_bbox_matches(grid, rectangle, int(color)):
            return None

    expected = _expected_output(grid.shape, rectangles)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None

    cover = _coverage_map(grid.shape, rectangles)
    max_cover = int(cover.max())
    if branch < 4 and max_cover != branch:
        return None
    if branch == 4 and max_cover < 4:
        return None
    if not np.any(cover == 1):
        return None
    if not np.any(cover == 0):
        return None
    for depth in range(2, min(max_cover, 3) + 1):
        if not np.any(cover == depth):
            return None
    if max_cover >= 4 and not np.any(cover >= 4):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        sample = _build_sample(rng)
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid layered-rectangle sample")
