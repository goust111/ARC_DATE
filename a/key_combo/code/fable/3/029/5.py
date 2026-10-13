from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
PINK = 6
CONTROL_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]


@dataclass(frozen=True)
class RaySpec:
    color: int
    control_col: int
    pillar_col: int
    height: int

    @property
    def direction(self) -> int:
        return 1 if self.pillar_col > self.control_col else -1

    @property
    def distance(self) -> int:
        return abs(self.pillar_col - self.control_col)


def _ray_cells(shape: tuple[int, int], control_col: int, pillar_col: int, pillar_height: int) -> list[tuple[int, int]]:
    h_grid, w_grid = shape
    assert 0 <= control_col < w_grid
    assert 0 <= pillar_col < w_grid
    assert control_col != pillar_col
    step = int(pillar_height)
    assert step >= 1

    pillar_top = h_grid - step
    stop_row = pillar_top - 1
    assert stop_row >= 1, "pillar must leave room for a ray above it"

    direction = 1 if pillar_col > control_col else -1
    distance = abs(pillar_col - control_col)
    assert (stop_row - 1) // step >= distance, "ray must reach the pillar column before stopping"

    cells: list[tuple[int, int]] = []
    for r in range(1, stop_row + 1):
        drift = min(distance, (r - 1) // step)
        c = control_col + direction * drift
        assert 0 <= c < w_grid
        cells.append((r, c))
    assert cells[-1][1] == pillar_col
    return cells


def _extract_specs(g: np.ndarray) -> list[RaySpec]:
    h, w = g.shape
    assert h >= 6 and w >= 3
    assert PINK not in set(map(int, g.ravel())), "pink is reserved for ray crossings"

    controls = [(int(c), int(g[0, c])) for c in range(w) if int(g[0, c]) != BG]
    assert len(controls) >= 2, "expected at least two top-row control pixels"
    colors = [color for _c, color in controls]
    assert len(set(colors)) == len(colors), "control colors must be distinct"

    all_colors = set(map(int, g.ravel())) - {BG}
    assert all_colors == set(colors), "every foreground color must have a top control pixel"

    specs: list[RaySpec] = []
    for control_col, color in controls:
        coords = np.argwhere(g == color)
        bottom_cols = [c for c in range(w) if int(g[h - 1, c]) == color]
        assert len(bottom_cols) == 1, "each color must have one bottom-anchored pillar"
        pillar_col = int(bottom_cols[0])
        assert pillar_col != control_col

        height = 0
        r = h - 1
        while r >= 0 and int(g[r, pillar_col]) == color:
            height += 1
            r -= 1
        assert height >= 1

        expected = {(0, control_col)}
        expected.update((rr, pillar_col) for rr in range(h - height, h))
        actual = {(int(rr), int(cc)) for rr, cc in coords}
        assert actual == expected, "a color must appear only as its control pixel and its vertical pillar"

        specs.append(RaySpec(int(color), int(control_col), int(pillar_col), int(height)))

    heights = [spec.height for spec in specs]
    assert len(set(heights)) == len(heights), "pillar heights must be different"
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace stepped downward rays from top controls, using same-color pillar heights as step periods.

    Input structure:
    - The grid has a black background with several distinct-colored control pixels on the top row.
    - For each control color, exactly one same-color vertical pillar stands on the bottom edge.
    - Pillar columns differ from their matching control columns, and pillar heights are all different.

    Rule:
    - For a pillar of height h, start the ray one row below the control pixel.
    - After every full block of h downward rows, drift one column toward the same-color pillar; after reaching that column, continue straight down.
    - Stop on the cell directly above the pillar top. Cells visited by two or more different-colored rays become pink, while controls and pillars remain unchanged.

    Key insight:
    - The pillar is both the parameter and the target: its height gives the ray period, and its side tells the ray which way to drift.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert np.all((0 <= g) & (g <= 9))

    specs = _extract_specs(g)
    visits: dict[tuple[int, int], set[int]] = {}
    for spec in specs:
        for r, c in _ray_cells(g.shape, spec.control_col, spec.pillar_col, spec.height):
            assert int(g[r, c]) == BG, "rays may cross only background cells in the input"
            visits.setdefault((r, c), set()).add(spec.color)

    out = g.copy()
    for (r, c), colors in visits.items():
        out[r, c] = PINK if len(colors) >= 2 else next(iter(colors))

    out[g != BG] = g[g != BG]
    return out


# ----------------------------- generator -----------------------------

_MAX_ATTEMPTS = 1200
_HEIGHT_OPTIONS = [2, 3, 4, 5, 6, 7]


def _paint_input(shape: tuple[int, int], specs: list[RaySpec]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    for spec in specs:
        grid[0, spec.control_col] = spec.color
        grid[h - spec.height : h, spec.pillar_col] = spec.color
    return grid


def _expected_output(inp: np.ndarray, specs: list[RaySpec]) -> np.ndarray | None:
    visits: dict[tuple[int, int], set[int]] = {}
    for spec in specs:
        try:
            cells = _ray_cells(inp.shape, spec.control_col, spec.pillar_col, spec.height)
        except AssertionError:
            return None
        for r, c in cells:
            if int(inp[r, c]) != BG:
                return None
            visits.setdefault((r, c), set()).add(spec.color)

    out = inp.copy()
    for (r, c), colors in visits.items():
        out[r, c] = PINK if len(colors) >= 2 else next(iter(colors))
    out[inp != BG] = inp[inp != BG]
    return out


def _well_spaced(value: int, used: set[int], gap: int = 1) -> bool:
    return all(abs(value - old) > gap for old in used)


def _choose_position(
    width: int,
    height: int,
    direction: int,
    used_controls: set[int],
    used_pillars: set[int],
) -> tuple[int, int] | None:
    max_distance = min(7, (height - 3) // 2)
    distances = list(range(1, max_distance + 1))
    random.shuffle(distances)
    for distance in distances:
        controls = list(range(width))
        random.shuffle(controls)
        for control_col in controls:
            pillar_col = control_col + direction * distance
            if not (0 <= pillar_col < width):
                continue
            if not _well_spaced(control_col, used_controls):
                continue
            if not _well_spaced(pillar_col, used_pillars):
                continue
            if pillar_col in used_controls or control_col in used_pillars:
                continue
            return int(control_col), int(pillar_col)
    return None


def _random_specs(shape: tuple[int, int], count: int) -> list[RaySpec] | None:
    h_grid, w_grid = shape
    colors = random.sample(CONTROL_COLORS, count)
    heights = random.sample([h for h in _HEIGHT_OPTIONS if h <= (h_grid - 4) // 2], count)
    if len(heights) != count:
        return None

    signs = [1, -1]
    while len(signs) < count:
        signs.append(random.choice([-1, 1]))
    random.shuffle(signs)

    used_controls: set[int] = set()
    used_pillars: set[int] = set()
    specs: list[RaySpec] = []
    for color, height, sign in zip(colors, heights, signs):
        pos = _choose_position(w_grid, h_grid, sign, used_controls, used_pillars)
        if pos is None:
            return None
        control_col, pillar_col = pos
        max_dist = (h_grid - height - 3) // height
        if abs(pillar_col - control_col) > max_dist:
            return None
        used_controls.add(control_col)
        used_pillars.add(pillar_col)
        specs.append(RaySpec(int(color), int(control_col), int(pillar_col), int(height)))

    return specs


def _overlap_stats(shape: tuple[int, int], specs: list[RaySpec]) -> tuple[int, int] | None:
    visits: dict[tuple[int, int], set[int]] = {}
    total = 0
    for spec in specs:
        try:
            cells = _ray_cells(shape, spec.control_col, spec.pillar_col, spec.height)
        except AssertionError:
            return None
        total += len(cells)
        for cell in cells:
            visits.setdefault(cell, set()).add(spec.color)
    overlap = sum(1 for colors in visits.values() if len(colors) >= 2)
    return overlap, total


def _semantically_rich(shape: tuple[int, int], specs: list[RaySpec], require_overlap: bool) -> bool:
    if len({spec.height for spec in specs}) != len(specs):
        return False
    if not any(spec.direction == 1 for spec in specs) or not any(spec.direction == -1 for spec in specs):
        return False
    if not any(spec.distance >= 2 for spec in specs):
        return False

    h_grid, _w_grid = shape
    for spec in specs:
        first_pillar_col_row = spec.distance * spec.height + 1
        stop_row = h_grid - spec.height - 1
        if stop_row - first_pillar_col_row + 1 < 2:
            return False

    stats = _overlap_stats(shape, specs)
    if stats is None:
        return False
    overlap, total = stats
    if require_overlap and overlap == 0:
        return False
    if not require_overlap and overlap != 0:
        return False
    if overlap > max(5, total // 7):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 30)
    w = random.randint(17, 30)
    count = random.choice([3, 3, 4])
    require_overlap = random.random() < 0.75

    specs = _random_specs((h, w), count)
    if specs is None:
        return None
    if not _semantically_rich((h, w), specs, require_overlap):
        return None

    inp = _paint_input((h, w), specs)
    expected = _expected_output(inp, specs)
    if expected is None:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stepped-ray input")
