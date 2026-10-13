from collections import Counter, deque

import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

_RNG = np.random.default_rng()

# Generator configuration.
GRID_H_RANGE = (18, 27)  # high is exclusive
GRID_W_RANGE = (18, 27)
NUM_CROSSES_RANGE = (2, 5)
ARM_MAX = 6
PLACEMENT_RETRIES = 120
GEN_RETRIES = 300
SEPARATION_PADDING = 1


def _corner_legend(g: np.ndarray) -> dict[str, int]:
    return {
        "up": int(g[0, 0]),
        "right": int(g[0, -1]),
        "down": int(g[-1, 0]),
        "left": int(g[-1, -1]),
    }


def _background_color(g: np.ndarray) -> int:
    mask = np.ones(g.shape, dtype=bool)
    mask[0, 0] = False
    mask[0, -1] = False
    mask[-1, 0] = False
    mask[-1, -1] = False
    values = g[mask]
    assert values.size > 0
    return int(Counter(map(int, values)).most_common(1)[0][0])


def _cross_color(g: np.ndarray, bg: int, legend: dict[str, int]) -> int:
    corner_colors = set(legend.values())
    candidates = [
        int(color)
        for color in np.unique(g)
        if int(color) != bg and int(color) not in corner_colors
    ]
    assert len(candidates) == 1
    return candidates[0]


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if visited[r, c]:
            continue
        visited[r, c] = True
        q = deque([(r, c)])
        comp: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            comp.append((rr, cc))
            for dr, dc in DIRS.values():
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    q.append((nr, nc))
        components.append(comp)
    return components


def _parse_cross(comp: list[tuple[int, int]]) -> dict[str, object]:
    cells = set(comp)
    centers = []
    for r, c in comp:
        if all((r + dr, c + dc) in cells for dr, dc in DIRS.values()):
            centers.append((r, c))
    assert len(centers) == 1
    cr, cc = centers[0]

    lengths: dict[str, int] = {}
    for name, (dr, dc) in DIRS.items():
        length = 0
        rr = cr + dr
        cc2 = cc + dc
        while (rr, cc2) in cells:
            length += 1
            rr += dr
            cc2 += dc
        assert length > 0
        lengths[name] = length

    assert len(set(lengths.values())) == 4

    expected = {(cr, col) for col in range(cc - lengths["left"], cc + lengths["right"] + 1)}
    expected |= {(row, cc) for row in range(cr - lengths["up"], cr + lengths["down"] + 1)}
    assert expected == cells

    shortest_dir = min(lengths, key=lengths.get)
    return {
        "center": (cr, cc),
        "lengths": lengths,
        "bbox": (
            cr - lengths["up"],
            cc - lengths["left"],
            cr + lengths["down"],
            cc + lengths["right"],
        ),
        "shortest_dir": shortest_dir,
    }


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= 3 and g.shape[1] >= 3

    legend = _corner_legend(g)
    assert len(set(legend.values())) == 4
    bg = _background_color(g)
    fg = _cross_color(g, bg, legend)

    components = _find_components(g == fg)
    assert components

    out = g.copy()
    for comp in components:
        cross = _parse_cross(comp)
        r0, c0, r1, c1 = cross["bbox"]
        fill = legend[cross["shortest_dir"]]
        out[r0 : r1 + 1, c0 : c1 + 1] = fill

    out[0, 0] = g[0, 0]
    out[0, -1] = g[0, -1]
    out[-1, 0] = g[-1, 0]
    out[-1, -1] = g[-1, -1]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each unequal-arm cross with its bounding rectangle.

    The rectangle color is chosen from the four corner legend pixels:
    top-left=up, top-right=right, bottom-left=down, bottom-right=left.
    The shortest arm direction selects which legend color to use.
    """
    return _apply_rule(grid)


def _paint_cross(grid: np.ndarray, center: tuple[int, int], lengths: dict[str, int], color: int) -> None:
    cr, cc = center
    grid[cr, cc - lengths["left"] : cc + lengths["right"] + 1] = color
    grid[cr - lengths["up"] : cr + lengths["down"] + 1, cc] = color


def _choose_lengths(limit: int) -> dict[str, int] | None:
    if limit < 4:
        return None
    values = _RNG.choice(np.arange(1, limit + 1), size=4, replace=False)
    values = [int(v) for v in values]
    _RNG.shuffle(values)
    return dict(zip(("up", "down", "left", "right"), values))


def _expanded_slice(lo: int, hi: int, limit: int) -> slice:
    start = max(0, lo - SEPARATION_PADDING)
    stop = min(limit, hi + SEPARATION_PADDING + 1)
    return slice(start, stop)


def _build_expected(grid: np.ndarray, specs: list[dict[str, object]], legend: dict[str, int]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        fill = legend[spec["shortest_dir"]]
        out[r0 : r1 + 1, c0 : c1 + 1] = fill
    out[0, 0] = grid[0, 0]
    out[0, -1] = grid[0, -1]
    out[-1, 0] = grid[-1, 0]
    out[-1, -1] = grid[-1, -1]
    return out


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(*GRID_H_RANGE))
    w = int(_RNG.integers(*GRID_W_RANGE))
    palette = [int(x) for x in _RNG.choice(np.arange(10), size=6, replace=False)]
    bg, fg, up_color, right_color, down_color, left_color = palette
    legend = {
        "up": up_color,
        "right": right_color,
        "down": down_color,
        "left": left_color,
    }

    grid = np.full((h, w), bg, dtype=int)
    grid[0, 0] = up_color
    grid[0, -1] = right_color
    grid[-1, 0] = down_color
    grid[-1, -1] = left_color

    blocked = np.zeros((h, w), dtype=bool)
    blocked[0, 0] = True
    blocked[0, -1] = True
    blocked[-1, 0] = True
    blocked[-1, -1] = True

    specs: list[dict[str, object]] = []
    target = int(_RNG.integers(*NUM_CROSSES_RANGE))
    arm_limit = min(ARM_MAX, h - 4, w - 4)

    for _ in range(target):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            lengths = _choose_lengths(arm_limit)
            if lengths is None:
                return None

            r_min = 1 + lengths["up"]
            r_max = (h - 2) - lengths["down"]
            c_min = 1 + lengths["left"]
            c_max = (w - 2) - lengths["right"]
            if r_min > r_max or c_min > c_max:
                continue

            cr = int(_RNG.integers(r_min, r_max + 1))
            cc = int(_RNG.integers(c_min, c_max + 1))
            r0 = cr - lengths["up"]
            r1 = cr + lengths["down"]
            c0 = cc - lengths["left"]
            c1 = cc + lengths["right"]

            rs = _expanded_slice(r0, r1, h)
            cs = _expanded_slice(c0, c1, w)
            if blocked[rs, cs].any():
                continue

            _paint_cross(grid, (cr, cc), lengths, fg)
            blocked[rs, cs] = True
            specs.append(
                {
                    "center": (cr, cc),
                    "lengths": lengths,
                    "bbox": (r0, c0, r1, c1),
                    "shortest_dir": min(lengths, key=lengths.get),
                }
            )
            placed = True
            break
        if not placed:
            return None

    if len(specs) < 2:
        return None

    expected = _build_expected(grid, specs, legend)
    if not np.array_equal(_apply_rule(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after retries")
