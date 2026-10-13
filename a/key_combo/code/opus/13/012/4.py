import random
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
CONTROL_PALETTE = (BLUE, RED, GREEN)
MAX_GEN_ATTEMPTS = 400


Coord = tuple[int, int]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components_from_cells(cells: set[Coord], connectivity: int) -> list[set[Coord]]:
    remaining = set(cells)
    out: list[set[Coord]] = []
    nbr_fn = _neighbors4 if connectivity == 4 else _neighbors8
    while remaining:
        seed = next(iter(remaining))
        stack = [seed]
        remaining.remove(seed)
        comp = {seed}
        while stack:
            r, c = stack.pop()
            for nr, nc in nbr_fn(r, c):
                nb = (nr, nc)
                if nb in remaining:
                    remaining.remove(nb)
                    comp.add(nb)
                    stack.append(nb)
        out.append(comp)
    return out


def _components_from_mask(mask: np.ndarray, connectivity: int) -> list[set[Coord]]:
    cells = {tuple(map(int, p)) for p in np.argwhere(mask)}
    return _components_from_cells(cells, connectivity)


def _normalize(cells: list[Coord] | set[Coord]) -> list[Coord]:
    pts = sorted((int(r), int(c)) for r, c in cells)
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return [(r - min_r, c - min_c) for r, c in pts]


def _mirror_h(cells: list[Coord]) -> list[Coord]:
    width = max(c for _, c in cells) + 1
    return _normalize([(r, width - 1 - c) for r, c in cells])


def _shape_size(cells: list[Coord]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return max(rows) + 1, max(cols) + 1


def _classify_region(region: set[Coord], row_to_color: dict[int, int]) -> bool:
    color_groups: dict[int, set[Coord]] = defaultdict(set)
    for r, c in region:
        color_groups[int(row_to_color[r])].add((r, c))

    subregions: list[set[Coord]] = []
    for cells in color_groups.values():
        subregions.extend(_components_from_cells(cells, connectivity=4))

    if len(subregions) < 2:
        return False

    graph = [set() for _ in subregions]
    for i in range(len(subregions)):
        for j in range(i + 1, len(subregions)):
            has_orth = False
            has_diag = False
            for r0, c0 in subregions[i]:
                for r1, c1 in subregions[j]:
                    if abs(r0 - r1) + abs(c0 - c1) == 1:
                        has_orth = True
                    if abs(r0 - r1) == 1 and abs(c0 - c1) == 1:
                        has_diag = True
                if has_orth:
                    break
            if has_orth:
                return False
            if has_diag:
                graph[i].add(j)
                graph[j].add(i)

    degrees = [len(nbrs) for nbrs in graph]
    if max(degrees, default=0) > 2:
        return False

    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for nb in graph[node]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    if len(seen) != len(subregions):
        return False

    edge_count = sum(degrees) // 2
    return edge_count == len(subregions) - 1


def main(grid: np.ndarray) -> np.ndarray:
    """Reveal only the row-recolored grey regions whose color-split lobes form a diagonal chain.

    Input structure:
    - The grid is a black background with several grey 8-connected regions that do not touch each other.
    - The rightmost column is a control column: each row that contains grey pixels has one colored control pixel there, while inactive rows stay black.
    - Grey regions may span multiple rows and become multi-colored after row-wise recoloring.

    Rule:
    - Read the control color from the rightmost column of each grey-containing row and recolor all grey pixels in that row with it.
    - For each original grey 8-connected region, split its recolored pixels into monochromatic 4-connected subregions.
    - Keep the recolored region only when those subregions touch one another exclusively by diagonal corners and their diagonal-contact graph is a connected, non-branching chain; otherwise erase the whole region to black.
    - The control column itself always remains unchanged.

    Key insight:
    - The control column does more than assign visible colors: it changes how one grey region breaks into color-based lobes, and only regions whose induced lobe graph is a pure diagonal chain survive.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    h, w = g.shape
    control_col = w - 1
    grey_mask = g == GREY
    assert not np.any(grey_mask[:, control_col]), "grey pixels may not occupy the control column"

    row_to_color: dict[int, int] = {}
    for r in range(h):
        grey_in_row = bool(np.any(grey_mask[r]))
        ctrl = int(g[r, control_col])
        if grey_in_row:
            assert ctrl not in (BLACK, GREY), "each grey row needs a non-black control color"
            row_to_color[r] = ctrl
        else:
            assert ctrl == BLACK, "rows without grey must have black in the control column"

    out = g.copy()
    out[grey_mask] = BLACK
    for region in _components_from_mask(grey_mask, connectivity=8):
        if _classify_region(region, row_to_color):
            for r, c in region:
                out[r, c] = row_to_color[r]
    return out


def _random_control_color(exclude: int | None = None) -> int:
    colors = list(CONTROL_PALETTE)
    if exclude in colors and len(colors) > 1:
        colors.remove(exclude)
    return random.choice(colors)


def _make_pass_diag() -> tuple[list[Coord], list[int], str]:
    length = random.randint(5, 7)
    col = 0
    cols = [col]
    for _ in range(length - 1):
        col += random.choice((-1, 1))
        cols.append(col)
    min_c = min(cols)
    cells = [(r, c - min_c) for r, c in enumerate(cols)]
    colors = [_random_control_color() for _ in range(length)]
    return cells, colors, "pass"


def _make_pass_sensitive() -> tuple[list[Coord], list[int], str]:
    cells = [(0, 0), (1, 1), (2, 1), (3, 2), (4, 3)]
    if random.random() < 0.5:
        cells = _mirror_h(cells)
    shared = _random_control_color()
    colors = [
        _random_control_color(),
        shared,
        shared,
        _random_control_color(exclude=shared),
        _random_control_color(),
    ]
    return cells, colors, "pass_sensitive"


def _make_fail_blob() -> tuple[list[Coord], list[int], str]:
    options = [
        [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
        [(0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 1)],
    ]
    cells = random.choice(options)
    if random.random() < 0.5:
        cells = _mirror_h(cells)
    row_count = max(r for r, _ in cells) + 1
    colors = [_random_control_color() for _ in range(row_count)]
    return cells, colors, "fail"


def _make_fail_stair() -> tuple[list[Coord], list[int], str]:
    cells = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4)]
    if random.random() < 0.5:
        cells = _mirror_h(cells)
    shared = _random_control_color()
    colors = [
        _random_control_color(exclude=shared),
        shared,
        shared,
        _random_control_color(exclude=shared),
    ]
    return cells, colors, "fail_same_color"


def _make_pass_domino_chain() -> tuple[list[Coord], list[int], str]:
    rows = random.randint(3, 4)
    cells = []
    for r in range(rows):
        cells.append((r, 2 * r))
        cells.append((r, 2 * r + 1))
    cells = _normalize(cells)
    if random.random() < 0.5:
        cells = _mirror_h(cells)
    colors = [_random_control_color() for _ in range(rows)]
    return cells, colors, "pass"


def _make_fail_sensitive() -> tuple[list[Coord], list[int], str]:
    cells = [(0, 0), (1, 1), (2, 1), (3, 2), (4, 3)]
    if random.random() < 0.5:
        cells = _mirror_h(cells)
    colors = []
    while len(colors) < 5:
        candidate = [_random_control_color() for _ in range(5)]
        if candidate[1] != candidate[2]:
            colors = candidate
    return cells, colors, "fail_sensitive"


def _build_region_specs() -> list[tuple[list[Coord], list[int], str]]:
    specs = [
        _make_pass_diag(),
        _make_pass_sensitive(),
        _make_fail_blob(),
        _make_fail_stair(),
    ]
    if random.random() < 0.7:
        extra = _make_pass_domino_chain() if random.random() < 0.6 else _make_fail_sensitive()
        specs.append(extra)
    random.shuffle(specs)
    return specs


def _place_specs(specs: list[tuple[list[Coord], list[int], str]]) -> np.ndarray | None:
    heights = [_shape_size(cells)[0] for cells, _, _ in specs]
    widths = [_shape_size(cells)[1] for cells, _, _ in specs]
    gap_choices = [random.randint(1, 2) for _ in range(len(specs) - 1)]
    top_pad = random.randint(1, 2)
    bottom_pad = random.randint(1, 2)
    needed_h = top_pad + sum(heights) + sum(gap_choices) + bottom_pad
    if needed_h > 28:
        return None

    h = needed_h + random.randint(0, min(2, 29 - needed_h))
    min_w = max(15, max(widths) + 5)
    max_w = min(24, max(widths) + 10)
    if min_w > max_w:
        return None
    w = random.randint(min_w, max_w)

    grid = np.zeros((h, w), dtype=int)
    regions_abs: list[set[Coord]] = []
    row_to_color: dict[int, int] = {}
    cursor = top_pad
    for idx, (cells, colors, _kind) in enumerate(specs):
        shape_h, shape_w = _shape_size(cells)
        max_left = w - 2 - shape_w
        if max_left < 1:
            return None
        left = random.randint(1, max_left)
        abs_cells = {(cursor + r, left + c) for r, c in cells}
        regions_abs.append(abs_cells)
        for r, c in abs_cells:
            grid[r, c] = GREY
        for rel_r, color in enumerate(colors):
            row_to_color[cursor + rel_r] = color
        cursor += shape_h
        if idx < len(specs) - 1:
            cursor += gap_choices[idx]

    for r, color in row_to_color.items():
        grid[r, -1] = color

    grey_components = _components_from_mask(grid == GREY, connectivity=8)
    if len(grey_components) != len(specs):
        return None
    if any(not (5 <= len(comp) <= 15) for comp in grey_components):
        return None

    pass_count = sum(_classify_region(comp, row_to_color) for comp in grey_components)
    fail_count = len(grey_components) - pass_count
    if pass_count < 2 or fail_count < 1:
        return None

    if not any(
        kind in {"fail_same_color", "fail_sensitive"}
        for (_cells, _colors, kind) in specs
    ):
        return None

    out = main(grid)
    if not np.array_equal(out[:, -1], grid[:, -1]):
        return None
    if np.count_nonzero(out == GREY) != 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        specs = _build_region_specs()
        grid = _place_specs(specs)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
