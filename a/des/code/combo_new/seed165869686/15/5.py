import numpy as np


BG_COLOR = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400
MAX_GRID_SIDE = 30
RNG = np.random.default_rng()

ARROW_SPECS = {
    "up": {
        "vec": (-1, 0),
        "offsets": {(0, 0), (-1, -1), (-1, 0), (-1, 1), (1, 0)},
    },
    "down": {
        "vec": (1, 0),
        "offsets": {(0, 0), (1, -1), (1, 0), (1, 1), (-1, 0)},
    },
    "left": {
        "vec": (0, -1),
        "offsets": {(0, 0), (-1, -1), (0, -1), (1, -1), (0, 1)},
    },
    "right": {
        "vec": (0, 1),
        "offsets": {(0, 0), (-1, 1), (0, 1), (1, 1), (0, -1)},
    },
}


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components8(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []

    for r_raw, c_raw in np.argwhere(mask):
        r = int(r_raw)
        c = int(c_raw)
        if visited[r, c]:
            continue

        stack = [(r, c)]
        visited[r, c] = True
        coords = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if not mask[nr, nc] or visited[nr, nc]:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

        components.append(np.array(coords, dtype=int))

    return components


def _identify_arrow(component: np.ndarray) -> tuple[tuple[int, int], str]:
    coords = {tuple(map(int, cell)) for cell in component.tolist()}
    matches = []
    for r, c in coords:
        for direction, spec in ARROW_SPECS.items():
            expected = {(r + dr, c + dc) for dr, dc in spec["offsets"]}
            if expected == coords:
                matches.append(((r, c), direction))

    assert len(matches) == 1, "expected exactly one arrow center/direction"
    return matches[0]


def _line_pixels_ahead(
    coords: np.ndarray, center: tuple[int, int], direction: str
) -> np.ndarray:
    if coords.size == 0:
        return coords

    r0, c0 = center
    dr, dc = ARROW_SPECS[direction]["vec"]
    if dr == 0:
        mask = coords[:, 0] == r0
        if dc > 0:
            mask &= coords[:, 1] > c0
        else:
            mask &= coords[:, 1] < c0
    else:
        mask = coords[:, 1] == c0
        if dr > 0:
            mask &= coords[:, 0] > r0
        else:
            mask &= coords[:, 0] < r0
    return coords[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull each arrow color's scattered forward pixels into a compact run beside the arrow.

    Input structure:
    - The background is black.
    - Each foreground color contains one 5-pixel monochrome arrow sprite whose center,
      3-pixel tip, and 1-pixel tail encode one of the four cardinal directions.
    - Extra pixels of that same color appear as isolated cells on the arrow's row or
      column, only on the side the arrow points toward.

    Rule:
    - For each color, identify the unique arrow component and recover its direction.
    - Count all same-color pixels on the arrow's forward row or forward column, then
      erase those scattered pixels.
    - Starting immediately beyond the arrow tip, draw that many same-color pixels as a
      continuous 1-cell-wide segment along the arrow direction.
    - Leave every arrow sprite unchanged and keep the grid size fixed.

    Key insight:
    - The distant same-color singletons are not noise: their count determines the
      length of the compact bar that gets attached to the arrow's tip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    out = g.copy()

    for color_raw in np.unique(g):
        color = int(color_raw)
        if color == bg:
            continue

        mask = g == color
        components = _components8(mask)
        arrow_component = None
        center = None
        direction = None
        for component in components:
            if len(component) != 5:
                continue
            try:
                center, direction = _identify_arrow(component)
            except AssertionError:
                continue
            arrow_component = component
            break

        assert arrow_component is not None and center is not None and direction is not None
        arrow_cells = {tuple(map(int, cell)) for cell in arrow_component.tolist()}

        residual = np.array(
            [cell for cell in np.argwhere(mask) if tuple(map(int, cell)) not in arrow_cells],
            dtype=int,
        )
        ahead = _line_pixels_ahead(residual, center, direction)
        if ahead.size:
            out[ahead[:, 0], ahead[:, 1]] = bg

        dr, dc = ARROW_SPECS[direction]["vec"]
        for step in range(2, 2 + len(ahead)):
            rr = center[0] + step * dr
            cc = center[1] + step * dc
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]
            assert int(out[rr, cc]) in (bg, color)
            out[rr, cc] = color

    return out


def _sample_spaced_positions(length: int, count: int) -> list[int]:
    assert count >= 1
    assert length >= 2 * count - 1
    base = np.sort(RNG.choice(length - count + 1, size=count, replace=False))
    return [int(value + idx) for idx, value in enumerate(base)]


def _scatter_offsets(direction: str, count: int) -> list[tuple[int, int]]:
    span = 2 * count - 1 + int(RNG.integers(0, 4))
    slots = _sample_spaced_positions(span, count)
    dr, dc = ARROW_SPECS[direction]["vec"]
    return [((3 + slot) * dr, (3 + slot) * dc) for slot in slots]


def _bar_offsets(direction: str, count: int) -> list[tuple[int, int]]:
    dr, dc = ARROW_SPECS[direction]["vec"]
    return [((2 + idx) * dr, (2 + idx) * dc) for idx in range(count)]


def _inflate(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    expanded = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _candidate_centers(
    shape: tuple[int, int], offsets: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    drs = [dr for dr, _ in offsets]
    dcs = [dc for _, dc in offsets]
    r_min = -min(drs)
    r_max = shape[0] - 1 - max(drs)
    c_min = -min(dcs)
    c_max = shape[1] - 1 - max(dcs)
    if r_min > r_max or c_min > c_max:
        return []

    centers = [(r, c) for r in range(r_min, r_max + 1) for c in range(c_min, c_max + 1)]
    RNG.shuffle(centers)
    return centers


def _translate(
    center: tuple[int, int], offsets: set[tuple[int, int]]
) -> set[tuple[int, int]]:
    r0, c0 = center
    return {(r0 + dr, c0 + dc) for dr, dc in offsets}


def _place_arrow(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    direction: str,
    count: int,
) -> bool:
    arrow_offsets = set(ARROW_SPECS[direction]["offsets"])
    scatter_offsets = set(_scatter_offsets(direction, count))
    bar_offsets = set(_bar_offsets(direction, count))

    input_offsets = arrow_offsets | scatter_offsets
    relevant_offsets = input_offsets | bar_offsets

    for center in _candidate_centers(grid.shape, relevant_offsets):
        input_cells = _translate(center, input_offsets)
        relevant_cells = _translate(center, relevant_offsets)
        if any(blocked[r, c] for r, c in relevant_cells):
            continue
        if any(int(grid[r, c]) != BG_COLOR for r, c in input_cells):
            continue

        for r, c in input_cells:
            grid[r, c] = color
        for r, c in _inflate(relevant_cells, grid.shape):
            blocked[r, c] = True
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    num_arrows = int(RNG.integers(1, 5))
    side_low = 11 + 2 * num_arrows
    height = int(RNG.integers(side_low, MAX_GRID_SIDE + 1))
    width = int(RNG.integers(side_low, MAX_GRID_SIDE + 1))

    grid = np.full((height, width), BG_COLOR, dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)

    colors = list(RNG.choice(PALETTE, size=num_arrows, replace=False))
    directions = list(RNG.permutation(list(ARROW_SPECS))[:num_arrows])
    counts = [int(RNG.integers(1, 6)) for _ in range(num_arrows)]

    order = list(range(num_arrows))
    RNG.shuffle(order)
    for idx in order:
        if not _place_arrow(
            grid=grid,
            blocked=blocked,
            color=colors[idx],
            direction=directions[idx],
            count=counts[idx],
        ):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
