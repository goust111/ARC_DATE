import random

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _ranked_foreground_colors(grid: np.ndarray) -> list[int]:
    fg = np.asarray(grid, dtype=int)
    values = fg[fg != 0]
    assert values.size > 0, "expected at least one foreground color"

    colors, counts = np.unique(values, return_counts=True)
    assert len(set(map(int, counts))) == len(counts), "foreground color counts must be unique"
    order = np.argsort(-counts)
    return [int(colors[i]) for i in order]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse the foreground color distribution by frequency rank.

    Input structure:
    - Background is black.
    - There are 3-5 distinct foreground colors.
    - Each foreground color occupies a unique number of pixels.

    Rule:
    - Rank foreground colors by pixel count from most frequent to least frequent.
    - Swap the 1st and last ranked colors, the 2nd and second-last, and so on.
    - A swap means recoloring every pixel of one color to the other color's value.
    - If the number of colors is odd, the middle-ranked color stays unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    ranked = _ranked_foreground_colors(g)
    mapping = {color: ranked[-idx - 1] for idx, color in enumerate(ranked)}

    out = g.copy()
    for src, dst in mapping.items():
        out[g == src] = int(dst)
    return out


# --------------- generator ---------------

PALETTE = tuple(range(1, 10))
GRID_HEIGHT_RANGE = (15, 23)
GRID_WIDTH_RANGE = (15, 23)
COUNT_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 80
GROWTH_ATTEMPTS = 120
MAX_GENERATION_ATTEMPTS = 300


def _sample_num_colors() -> int:
    if random.random() < 0.5:
        return 4
    return random.choice((3, 5))


def _sample_counts(k: int, area: int) -> list[int] | None:
    total_cap = area // 3
    largest_cap = min(36, area // 5 + 8)

    for _ in range(COUNT_ATTEMPTS):
        smallest = random.randint(3, 10)
        counts = [smallest]
        for _ in range(k - 1):
            counts.append(counts[-1] + random.randint(2, 7))

        if counts[-1] > largest_cap:
            continue
        if sum(counts) > total_cap:
            continue
        return list(reversed(counts))
    return None


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr, dc in DIRS8:
        src_r0 = max(0, -dr)
        src_r1 = min(h, h - dr)
        src_c0 = max(0, -dc)
        src_c1 = min(w, w - dc)
        dst_r0 = src_r0 + dr
        dst_r1 = src_r1 + dr
        dst_c0 = src_c0 + dc
        dst_c1 = src_c1 + dc
        out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _grow_component(available: np.ndarray, size: int) -> np.ndarray | None:
    seeds = np.argwhere(available)
    if len(seeds) < size:
        return None

    h, w = available.shape

    def add_frontier(frontier: set[tuple[int, int]], selected: set[tuple[int, int]], r: int, c: int) -> None:
        for dr, dc in DIRS4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and available[rr, cc] and (rr, cc) not in selected:
                frontier.add((rr, cc))

    for _ in range(GROWTH_ATTEMPTS):
        seed = tuple(map(int, seeds[random.randrange(len(seeds))]))
        selected = {seed}
        frontier: set[tuple[int, int]] = set()
        add_frontier(frontier, selected, *seed)

        while len(selected) < size:
            if not frontier:
                break
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            selected.add(cell)
            add_frontier(frontier, selected, *cell)

        if len(selected) == size:
            return np.array(sorted(selected), dtype=int)

    return None


def _place_components(shape: tuple[int, int], counts: list[int]) -> list[np.ndarray] | None:
    occupied = np.zeros(shape, dtype=bool)
    placed: list[np.ndarray] = []

    for count in counts:
        component = None
        for _ in range(PLACEMENT_ATTEMPTS):
            blocked = _dilate8(occupied)
            available = ~blocked
            component = _grow_component(available, count)
            if component is not None:
                occupied[component[:, 0], component[:, 1]] = True
                placed.append(component)
                break
        if component is None:
            return None

    return placed


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    k = _sample_num_colors()
    counts = _sample_counts(k, height * width)
    if counts is None:
        return None

    components = _place_components((height, width), counts)
    if components is None:
        return None

    colors = random.sample(PALETTE, k)
    grid = np.zeros((height, width), dtype=int)
    for color, component in zip(colors, components):
        grid[component[:, 0], component[:, 1]] = int(color)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(out == 0, grid == 0):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
