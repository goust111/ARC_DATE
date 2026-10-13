import random

import numpy as np


BG_COLOR = 0
PALETTE = tuple(range(1, 10))

# Balance the odd/even branch so middle-rank "unchanged" cases appear regularly.
EVEN_COLOR_COUNTS = (2, 4)
ODD_COLOR_COUNTS = (3, 5)

HEIGHT_RANGE = (12, 19)
WIDTH_RANGE = (12, 19)

SHAPE_TYPES = ("rect", "bar", "hollow", "plus", "lshape")
MIN_COUNT_GAP = 2
SHAPE_SAMPLE_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 120
MAX_GENERATION_ATTEMPTS = 400


def _ranked_colors_by_frequency(grid: np.ndarray) -> list[int]:
    fg = grid[grid != BG_COLOR]
    colors, counts = np.unique(fg, return_counts=True)
    assert colors.size > 0, "expected at least one non-background color"

    count_list = [int(x) for x in counts]
    assert len(set(count_list)) == len(count_list), "non-background colors must have distinct counts"

    ranked = list(zip(count_list, [int(x) for x in colors]))
    ranked.sort(key=lambda item: -item[0])
    return [color for _count, color in ranked]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each non-background color with the color whose frequency rank is mirrored.

    Input structure:
    - Black (0) is the background.
    - Every non-background color has a distinct global pixel count.

    Rule:
    - Rank the non-background colors from most frequent to least frequent.
    - Swap rank 1 with rank N, rank 2 with rank N-1, and so on.
    - If N is odd, the middle-ranked color stays the same.
    - Background pixels are unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    ranked = _ranked_colors_by_frequency(g)
    mirrored = ranked[::-1]
    mapping = {src: dst for src, dst in zip(ranked, mirrored)}

    out = g.copy()
    for src, dst in mapping.items():
        out[g == src] = dst
    return out


def _solid_rect_mask() -> np.ndarray:
    h = random.randint(2, 5)
    w = random.randint(2, 6)
    return np.ones((h, w), dtype=bool)


def _bar_mask() -> np.ndarray:
    length = random.randint(3, 8)
    if random.choice((True, False)):
        return np.ones((1, length), dtype=bool)
    return np.ones((length, 1), dtype=bool)


def _hollow_rect_mask() -> np.ndarray:
    h = random.randint(3, 7)
    w = random.randint(3, 7)
    mask = np.ones((h, w), dtype=bool)
    mask[1:-1, 1:-1] = False
    return mask


def _plus_mask() -> np.ndarray:
    h = random.choice((3, 5, 7))
    w = random.choice((3, 5, 7))
    mask = np.zeros((h, w), dtype=bool)
    mask[:, w // 2] = True
    mask[h // 2, :] = True
    return mask


def _lshape_mask() -> np.ndarray:
    h = random.randint(3, 7)
    w = random.randint(3, 7)
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = True
    mask[-1, :] = True
    return mask


def _sample_shape_mask() -> np.ndarray:
    shape_type = random.choice(SHAPE_TYPES)
    if shape_type == "rect":
        return _solid_rect_mask()
    if shape_type == "bar":
        return _bar_mask()
    if shape_type == "hollow":
        return _hollow_rect_mask()
    if shape_type == "plus":
        return _plus_mask()
    return _lshape_mask()


def _sample_color_count() -> int:
    if random.choice((True, False)):
        return random.choice(EVEN_COLOR_COUNTS)
    return random.choice(ODD_COLOR_COUNTS)


def _sample_shapes(n_colors: int) -> list[dict[str, np.ndarray | int]] | None:
    shapes: list[dict[str, np.ndarray | int]] = []
    used_counts: list[int] = []

    for color in random.sample(PALETTE, n_colors):
        chosen = None
        for _ in range(SHAPE_SAMPLE_ATTEMPTS):
            mask = _sample_shape_mask()
            count = int(mask.sum())
            if any(abs(count - seen) < MIN_COUNT_GAP for seen in used_counts):
                continue
            chosen = {"color": int(color), "mask": mask, "count": count}
            used_counts.append(count)
            break
        if chosen is None:
            return None
        shapes.append(chosen)

    shapes.sort(key=lambda item: int(np.prod(item["mask"].shape)), reverse=True)
    return shapes


def _can_place(blocked: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = mask.shape
    return not np.any(blocked[top : top + h, left : left + w] & mask)


def _paint_shape(
    grid: np.ndarray,
    blocked: np.ndarray,
    mask: np.ndarray,
    color: int,
    top: int,
    left: int,
) -> None:
    h, w = mask.shape
    window = grid[top : top + h, left : left + w]
    window[mask] = color

    for dr, dc in np.argwhere(mask):
        r = int(top + dr)
        c = int(left + dc)
        blocked[max(0, r - 1) : min(grid.shape[0], r + 2), max(0, c - 1) : min(grid.shape[1], c + 2)] = True


def _place_shapes(height: int, width: int, shapes: list[dict[str, np.ndarray | int]]) -> np.ndarray | None:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    for item in shapes:
        mask = item["mask"]
        color = int(item["color"])
        h, w = mask.shape
        if h > height or w > width:
            return None

        positions = [(r, c) for r in range(height - h + 1) for c in range(width - w + 1)]
        random.shuffle(positions)

        placed = False
        for top, left in positions[:PLACEMENT_ATTEMPTS]:
            if not _can_place(blocked, mask, top, left):
                continue
            _paint_shape(grid, blocked, mask, color, top, left)
            placed = True
            break
        if not placed:
            return None

    return grid


def _count_foreground(grid: np.ndarray) -> dict[int, int]:
    counts: dict[int, int] = {}
    for value in grid.ravel():
        color = int(value)
        if color == BG_COLOR:
            continue
        counts[color] = counts.get(color, 0) + 1
    return counts


def _expected_output(grid: np.ndarray) -> np.ndarray:
    counts = _count_foreground(grid)
    assert counts, "generator must create at least one foreground color"
    assert len(set(counts.values())) == len(counts), "generator must create unique color counts"

    ranked = [color for color, _count in sorted(counts.items(), key=lambda item: item[1], reverse=True)]
    mapping = {src: dst for src, dst in zip(ranked, ranked[::-1])}

    out = grid.copy()
    for src, dst in mapping.items():
        out[grid == src] = dst
    return out


def _gen_attempt() -> np.ndarray | None:
    n_colors = _sample_color_count()
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)

    shapes = _sample_shapes(n_colors)
    if shapes is None:
        return None

    grid = _place_shapes(height, width, shapes)
    if grid is None:
        return None

    counts = _count_foreground(grid)
    if len(counts) != n_colors or len(set(counts.values())) != n_colors:
        return None

    expected = _expected_output(grid)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the frequency-rank recoloring task")
