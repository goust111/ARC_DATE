import random
from collections import Counter

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
BROWN = 9

DIR_ORDER = ("up", "right", "down", "left")
DIR_STEPS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
ANTENNA_OFFSETS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
SIGNAL_BASES = {
    BLUE: frozenset({BLUE}),
    RED: frozenset({RED}),
    GREEN: frozenset({GREEN}),
    YELLOW: frozenset({RED, GREEN}),
    MAGENTA: frozenset({RED, BLUE}),
    AZURE: frozenset({BLUE, GREEN}),
    BROWN: frozenset({RED, BLUE, GREEN}),
}
BASES_TO_COLOR = {
    frozenset({BLUE}): BLUE,
    frozenset({RED}): RED,
    frozenset({GREEN}): GREEN,
    frozenset({RED, BLUE}): MAGENTA,
    frozenset({RED, GREEN}): YELLOW,
    frozenset({BLUE, GREEN}): AZURE,
    frozenset({RED, BLUE, GREEN}): BROWN,
}
PRIMARY_COLORS = (BLUE, RED, GREEN)
GEN_RETRIES = 400


def _bg_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(counts.most_common(1)[0][0])


def _tower_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def _detect_towers(grid: np.ndarray, bg: int) -> list[dict]:
    height, width = grid.shape
    towers: list[dict] = []
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            core = int(grid[r, c])
            if core in (bg, GRAY, ORANGE):
                continue
            if core not in SIGNAL_BASES:
                continue

            patch = grid[r - 1 : r + 2, c - 1 : c + 2]
            if not np.isin(patch, [core, ORANGE]).all():
                continue

            corners = (
                int(grid[r - 1, c - 1]),
                int(grid[r - 1, c + 1]),
                int(grid[r + 1, c - 1]),
                int(grid[r + 1, c + 1]),
            )
            if any(color != core for color in corners):
                continue

            antennas = []
            ok = True
            for name, (dr, dc) in ANTENNA_OFFSETS.items():
                value = int(grid[r + dr, c + dc])
                if value == ORANGE:
                    antennas.append(name)
                elif value != core:
                    ok = False
                    break
            if ok and antennas:
                towers.append({"center": (r, c), "color": core, "antennas": tuple(antennas)})
    return towers


def _ray_cells(height: int, width: int, center: tuple[int, int], direction: str):
    r, c = center
    if direction == "up":
        for rr in range(r - 2, -1, -1):
            for cc in range(c - 1, c + 2):
                yield rr, cc
    elif direction == "down":
        for rr in range(r + 2, height):
            for cc in range(c - 1, c + 2):
                yield rr, cc
    elif direction == "left":
        for cc in range(c - 2, -1, -1):
            for rr in range(r - 1, r + 2):
                yield rr, cc
    else:
        assert direction == "right"
        for cc in range(c + 2, width):
            for rr in range(r - 1, r + 2):
                yield rr, cc


def _coverage_from_towers(height: int, width: int, towers: list[dict]) -> dict[tuple[int, int], set[int]]:
    coverage: dict[tuple[int, int], set[int]] = {}
    for tower in towers:
        bases = SIGNAL_BASES[int(tower["color"])]
        for direction in tower["antennas"]:
            for cell in _ray_cells(height, width, tower["center"], direction):
                coverage.setdefault(cell, set()).update(bases)
    return coverage


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend 3-pixel-wide signals from tower antennas, let gray mask cells ignore
    signal reception, and mix overlapping red/blue/green families by additive
    color union.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    bg = _bg_color(g)
    towers = _detect_towers(g, bg)
    assert towers, "expected at least one signal tower"

    tower_mask = np.zeros((height, width), dtype=bool)
    for tower in towers:
        for r, c in _tower_cells(tower["center"]):
            tower_mask[r, c] = True

    coverage = _coverage_from_towers(height, width, towers)

    out = g.copy()
    for (r, c), bases in coverage.items():
        if tower_mask[r, c] or int(g[r, c]) == GRAY:
            continue
        out[r, c] = int(BASES_TO_COLOR[frozenset(bases)])
    return out


def _ordered_antennas(required: tuple[str, ...]) -> tuple[str, ...]:
    chosen = set(required)
    for direction in DIR_ORDER:
        if direction in chosen:
            continue
        if random.random() < 0.35:
            chosen.add(direction)
    return tuple(direction for direction in DIR_ORDER if direction in chosen)


def _layout_pair_horizontal(height: int, width: int) -> list[dict] | None:
    if width < 13 or height < 9:
        return None
    row = random.randint(4, height - 5)
    left_col = random.randint(2, width - 9)
    right_col = random.randint(left_col + 6, width - 3)
    colors = random.sample(PRIMARY_COLORS, 2)
    return [
        {"center": (row, left_col), "color": colors[0], "antennas": _ordered_antennas(("right",))},
        {"center": (row, right_col), "color": colors[1], "antennas": _ordered_antennas(("left",))},
    ]


def _layout_pair_vertical(height: int, width: int) -> list[dict] | None:
    if height < 13 or width < 9:
        return None
    col = random.randint(4, width - 5)
    top_row = random.randint(2, height - 9)
    bottom_row = random.randint(top_row + 6, height - 3)
    colors = random.sample(PRIMARY_COLORS, 2)
    return [
        {"center": (top_row, col), "color": colors[0], "antennas": _ordered_antennas(("down",))},
        {"center": (bottom_row, col), "color": colors[1], "antennas": _ordered_antennas(("up",))},
    ]


def _layout_triple_cross(height: int, width: int) -> list[dict] | None:
    if height < 15 or width < 15:
        return None
    from_top = random.random() < 0.5
    if from_top:
        row = random.randint(6, height - 6)
    else:
        row = random.randint(5, height - 7)
    col = random.randint(6, width - 7)
    left_col = random.randint(2, col - 4)
    right_col = random.randint(col + 4, width - 3)
    colors = list(PRIMARY_COLORS)
    random.shuffle(colors)

    towers = [
        {"center": (row, left_col), "color": colors[0], "antennas": _ordered_antennas(("right",))},
        {"center": (row, right_col), "color": colors[1], "antennas": _ordered_antennas(("left",))},
    ]
    if from_top:
        top_row = random.randint(2, row - 4)
        towers.append({"center": (top_row, col), "color": colors[2], "antennas": _ordered_antennas(("down",))})
    else:
        bottom_row = random.randint(row + 4, height - 3)
        towers.append({"center": (bottom_row, col), "color": colors[2], "antennas": _ordered_antennas(("up",))})
    return towers


def _forbidden_near_towers(towers: list[dict], height: int, width: int) -> set[tuple[int, int]]:
    forbidden: set[tuple[int, int]] = set()
    for tower in towers:
        r, c = tower["center"]
        for rr in range(r - 2, r + 3):
            for cc in range(c - 2, c + 3):
                if 0 <= rr < height and 0 <= cc < width:
                    forbidden.add((rr, cc))
    return forbidden


def _build_connected_mask(
    height: int,
    width: int,
    signal_cells: list[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    valid_seeds = [cell for cell in signal_cells if cell not in forbidden and 1 <= cell[0] < height - 1 and 1 <= cell[1] < width - 1]
    if not valid_seeds:
        return None
    target_size = random.randint(5, 12)
    mask = {random.choice(valid_seeds)}
    attempts = 0
    while len(mask) < target_size and attempts < 300:
        base_r, base_c = random.choice(tuple(mask))
        neighbors = []
        for dr, dc in DIR_STEPS.values():
            rr = base_r + dr
            cc = base_c + dc
            if not (1 <= rr < height - 1 and 1 <= cc < width - 1):
                continue
            if (rr, cc) in forbidden:
                continue
            neighbors.append((rr, cc))
        if not neighbors:
            attempts += 1
            continue
        candidate = random.choice(neighbors)
        mask.add(candidate)
        attempts += 1
    if len(mask) < 4:
        return None
    return mask


def _draw_tower(grid: np.ndarray, tower: dict) -> None:
    r, c = tower["center"]
    color = int(tower["color"])
    grid[r - 1 : r + 2, c - 1 : c + 2] = color
    for direction in tower["antennas"]:
        dr, dc = ANTENNA_OFFSETS[direction]
        grid[r + dr, c + dc] = ORANGE


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 24)
    width = random.randint(16, 24)
    scenario = random.choice(("pair_h", "pair_v", "triple"))
    if scenario == "pair_h":
        towers = _layout_pair_horizontal(height, width)
    elif scenario == "pair_v":
        towers = _layout_pair_vertical(height, width)
    else:
        towers = _layout_triple_cross(height, width)
    if towers is None:
        return None

    tower_cells = {cell for tower in towers for cell in _tower_cells(tower["center"])}
    coverage = _coverage_from_towers(height, width, towers)
    non_tower_coverage = {cell: bases for cell, bases in coverage.items() if cell not in tower_cells}
    if not non_tower_coverage:
        return None

    overlap_sizes = [len(bases) for bases in non_tower_coverage.values()]
    if scenario == "triple":
        if 3 not in overlap_sizes:
            return None
    else:
        if 2 not in overlap_sizes:
            return None

    forbidden = _forbidden_near_towers(towers, height, width)
    signal_cells = [cell for cell in non_tower_coverage if len(non_tower_coverage[cell]) >= 1]
    mask_cells = _build_connected_mask(height, width, signal_cells, forbidden)
    if mask_cells is None:
        return None
    if sum(cell in non_tower_coverage for cell in mask_cells) < 2:
        return None

    grid = np.zeros((height, width), dtype=int)
    for tower in towers:
        _draw_tower(grid, tower)
    for r, c in mask_cells:
        grid[r, c] = GRAY

    out = main(grid)
    changed = int((out != grid).sum())
    if changed < 12:
        return None

    mixed_colors = {MAGENTA, YELLOW, AZURE}
    if scenario == "triple":
        if int((out == BROWN).sum()) == 0:
            return None
    else:
        if int(sum((out == color).sum() for color in mixed_colors)) == 0:
            return None

    if int((grid == GRAY).sum()) < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tower-signal input")
