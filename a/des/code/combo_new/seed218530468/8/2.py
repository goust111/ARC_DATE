from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

Coord = tuple[int, int]
Vec = tuple[int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _step(cell: Coord, direction: Vec, k: int = 1) -> Coord:
    return cell[0] + direction[0] * k, cell[1] + direction[1] * k


def _line_cells(a: Coord, b: Coord) -> list[Coord]:
    if a[0] == b[0]:
        c0, c1 = sorted((a[1], b[1]))
        return [(a[0], c) for c in range(c0, c1 + 1)]
    if a[1] == b[1]:
        r0, r1 = sorted((a[0], b[0]))
        return [(r, a[1]) for r in range(r0, r1 + 1)]
    raise AssertionError("line must be horizontal or vertical")


def _trace_pair(shape: tuple[int, int], a: Coord, da: Vec, b: Coord, db: Vec) -> set[Coord]:
    """Advance both endpoint rays together until they first share a row or column."""
    pa = a
    pb = b
    limit = shape[0] + shape[1]
    for _ in range(limit):
        pa = _step(pa, da)
        pb = _step(pb, db)
        assert _in_bounds(shape, *pa), "first ray leaves the grid before aligning"
        assert _in_bounds(shape, *pb), "second ray leaves the grid before aligning"
        if pa[0] == pb[0] or pa[1] == pb[1]:
            cells = set(_line_cells(a, pa))
            cells.update(_line_cells(b, pb))
            cells.update(_line_cells(pa, pb))
            return cells
    raise AssertionError("rays never align")


def _direction_from_marker(grid: np.ndarray, endpoint: Coord) -> Vec:
    found: list[Vec] = []
    for dr, dc in DIRS4:
        nr, nc = endpoint[0] + dr, endpoint[1] + dc
        if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == GREY:
            found.append((dr, dc))
    assert len(found) == 1, "each endpoint must have exactly one grey direction marker"
    return found[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color endpoint pair by following its adjacent grey direction markers and drawing the implied orthogonal path.

    Input structure:
    - The background is black.
    - Every non-black, non-grey color appears exactly twice, forming one endpoint pair.
    - Each endpoint has exactly one grey pixel directly above, below, left, or right of it, and that grey pixel indicates the endpoint's initial travel direction.

    Rule:
    - Remove all grey direction markers.
    - For each same-color endpoint pair, advance both endpoints one step at a time in the directions indicated by their markers.
    - When the two advancing rays first share a row or a column, connect their current positions with one final straight segment and paint the whole route in the endpoint color.

    Key insight:
    - The endpoints alone do not determine which orthogonal route to draw; the adjacent grey markers disambiguate the initial directions, and the path is completed at the first row/column alignment of the two directed rays.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = [int(color) for color in np.unique(g) if color not in (BACKGROUND, GREY)]
    out = np.full_like(g, BACKGROUND)

    for color in colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        assert len(coords) == 2, "each endpoint color must appear exactly twice"
        a, b = coords
        da = _direction_from_marker(g, a)
        db = _direction_from_marker(g, b)
        for r, c in _trace_pair(g.shape, a, da, b, db):
            out[r, c] = color

    return out


MIN_SIZE = 12
MAX_SIZE = 24
PAIR_ATTEMPTS = 200
GEN_ATTEMPTS = 400
MAX_SIDE = 7
MIN_ARM = 2


def _expand_cells(cells: set[Coord], shape: tuple[int, int]) -> set[Coord]:
    expanded: set[Coord] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if _in_bounds(shape, rr, cc):
                    expanded.add((rr, cc))
    return expanded


def _make_l(shape: tuple[int, int]) -> tuple[Coord, Vec, Coord, Vec, set[Coord]] | None:
    h, w = shape
    bend = (random.randrange(h), random.randrange(w))
    da = random.choice([(-1, 0), (1, 0)])
    db = random.choice([(0, -1), (0, 1)])

    max_a = bend[0] if da == (1, 0) else h - 1 - bend[0]
    max_b = bend[1] if db == (0, 1) else w - 1 - bend[1]
    max_a = min(max_a, MAX_SIDE)
    max_b = min(max_b, MAX_SIDE)
    if max_a < MIN_ARM or max_b < MIN_ARM:
        return None

    len_a = random.randint(MIN_ARM, max_a)
    len_b = random.randint(MIN_ARM, max_b)
    a = _step(bend, da, -len_a)
    b = _step(bend, db, -len_b)
    path = set(_line_cells(a, bend))
    path.update(_line_cells(b, bend))
    return a, da, b, db, path


def _make_z_horizontal(shape: tuple[int, int]) -> tuple[Coord, Vec, Coord, Vec, set[Coord]] | None:
    h, w = shape
    if h < 4 or w < 6:
        return None

    mid_c = random.randrange(w)
    max_side = min(mid_c, w - 1 - mid_c, MAX_SIDE)
    if max_side < MIN_ARM:
        return None

    side = random.randint(MIN_ARM, max_side)
    rows = random.sample(range(h), 2)
    if abs(rows[0] - rows[1]) < 2:
        return None
    r0, r1 = rows
    a = (r0, mid_c - side)
    b = (r1, mid_c + side)
    da = (0, 1)
    db = (0, -1)
    path = set(_line_cells(a, (r0, mid_c)))
    path.update(_line_cells((r0, mid_c), (r1, mid_c)))
    path.update(_line_cells((r1, mid_c), b))
    return a, da, b, db, path


def _make_z_vertical(shape: tuple[int, int]) -> tuple[Coord, Vec, Coord, Vec, set[Coord]] | None:
    h, w = shape
    if h < 6 or w < 4:
        return None

    mid_r = random.randrange(h)
    max_side = min(mid_r, h - 1 - mid_r, MAX_SIDE)
    if max_side < MIN_ARM:
        return None

    side = random.randint(MIN_ARM, max_side)
    cols = random.sample(range(w), 2)
    if abs(cols[0] - cols[1]) < 2:
        return None
    c0, c1 = cols
    a = (mid_r - side, c0)
    b = (mid_r + side, c1)
    da = (1, 0)
    db = (-1, 0)
    path = set(_line_cells(a, (mid_r, c0)))
    path.update(_line_cells((mid_r, c0), (mid_r, c1)))
    path.update(_line_cells((mid_r, c1), b))
    return a, da, b, db, path


def _sample_pair(kind: str, shape: tuple[int, int]) -> tuple[Coord, Vec, Coord, Vec, set[Coord]] | None:
    for _ in range(PAIR_ATTEMPTS):
        if kind == "L":
            spec = _make_l(shape)
        else:
            spec = _make_z_horizontal(shape) if random.random() < 0.5 else _make_z_vertical(shape)
        if spec is None:
            continue

        a, da, b, db, expected_path = spec
        if _step(a, da) == _step(b, db):
            continue
        if expected_path != _trace_pair(shape, a, da, b, db):
            continue
        return spec
    return None


def _pair_count(shape: tuple[int, int]) -> int:
    area = shape[0] * shape[1]
    max_pairs = min(5, 2 + area // 110, len(COLOR_POOL))
    return random.randint(2, max_pairs)


def _build_kind_list(count: int) -> list[str]:
    kinds = ["L", "Z"] * (count // 2)
    if count % 2 == 1:
        kinds.append(random.choice(["L", "Z"]))
    random.shuffle(kinds)
    return kinds


def _render_input(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        a, da = spec["a"], spec["da"]
        b, db = spec["b"], spec["db"]
        grid[a] = color
        grid[b] = color
        grid[_step(a, da)] = GREY
        grid[_step(b, db)] = GREY
    return grid


def _render_output(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        for r, c in spec["path"]:
            grid[r, c] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(MIN_SIZE, MAX_SIZE), random.randint(MIN_SIZE, MAX_SIZE))
    n_pairs = _pair_count(shape)
    colors = random.sample(COLOR_POOL, n_pairs)
    kinds = _build_kind_list(n_pairs)

    specs: list[dict[str, object]] = []
    blocked: set[Coord] = set()

    for color, kind in zip(colors, kinds):
        placed = False
        for _ in range(PAIR_ATTEMPTS):
            spec = _sample_pair(kind, shape)
            if spec is None:
                break
            a, da, b, db, path = spec
            if path & blocked:
                continue
            specs.append({"color": color, "a": a, "da": da, "b": b, "db": db, "path": path})
            blocked.update(_expand_cells(path, shape))
            placed = True
            break
        if not placed:
            return None

    inp = _render_input(shape, specs)
    expected = _render_output(shape, specs)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
