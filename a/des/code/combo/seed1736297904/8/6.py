from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6
ORANGE = 7

SEED_COLORS = (RED, BLUE, YELLOW)
MIX_COLORS = {
    frozenset((RED, BLUE)): MAGENTA,
    frozenset((RED, YELLOW)): ORANGE,
    frozenset((BLUE, YELLOW)): GREEN,
}

GRID_MIN = 10
GRID_MAX = 18
SEED_MARGIN = 1
MIN_SEGMENT_SPAN = 3
GEN_RETRIES = 400

Coord = tuple[int, int]


def _segment_cells(start: Coord, end: Coord) -> list[Coord]:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("segments must be horizontal or vertical")


def _extract_seed_pairs(grid: np.ndarray) -> tuple[dict[int, tuple[Coord, Coord]], set[Coord]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    colors = [int(color) for color in np.unique(g) if int(color) != BLACK]
    assert 1 <= len(colors) <= 3

    pairs: dict[int, tuple[Coord, Coord]] = {}
    all_seeds: set[Coord] = set()
    for color in colors:
        assert color in SEED_COLORS, "seed colors must be red, blue, or yellow"
        coords = np.argwhere(g == color)
        assert coords.shape[0] == 2, "each seed color must appear exactly twice"
        p0 = tuple(map(int, coords[0]))
        p1 = tuple(map(int, coords[1]))
        assert p0 != p1
        pairs[color] = (p0, p1)
        assert p0 not in all_seeds and p1 not in all_seeds
        all_seeds.add(p0)
        all_seeds.add(p1)
    return pairs, all_seeds


def _resolve_pair_path(p0: Coord, p1: Coord, all_seeds: set[Coord]) -> set[Coord]:
    own = {p0, p1}
    other_seeds = all_seeds - own

    if p0[0] == p1[0] or p0[1] == p1[1]:
        cells = set(_segment_cells(p0, p1))
        assert not (cells - own) & other_seeds, "straight path may not pass through another seed"
        return cells

    candidates: list[set[Coord]] = []
    for elbow in ((p0[0], p1[1]), (p1[0], p0[1])):
        seg_a = _segment_cells(p0, elbow)
        seg_b = _segment_cells(elbow, p1)
        cells = set(seg_a[:-1] + seg_b)
        if not (cells - own) & other_seeds:
            candidates.append(cells)

    assert len(candidates) == 1, "L-bend direction must be uniquely determined by other seeds"
    return candidates[0]


def _resolve_paths(grid: np.ndarray) -> tuple[dict[int, tuple[Coord, Coord]], dict[int, set[Coord]]]:
    pairs, all_seeds = _extract_seed_pairs(grid)
    paths = {
        int(color): _resolve_pair_path(p0, p1, all_seeds)
        for color, (p0, p1) in pairs.items()
    }
    return pairs, paths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color seed pairs with straight or L-shaped orthogonal paths.

    Different colors mix at crossings:
    - red + blue -> magenta
    - red + yellow -> orange
    - blue + yellow -> green

    Seed pixels stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    pairs, paths = _resolve_paths(g)

    seed_lookup = {
        coord: int(color)
        for color, endpoints in pairs.items()
        for coord in endpoints
    }
    contributors: dict[Coord, set[int]] = {}
    for color, cells in paths.items():
        for cell in cells:
            if cell in seed_lookup:
                assert seed_lookup[cell] == color
                continue
            r, c = cell
            assert int(g[r, c]) == BLACK, "non-seed path cells must start on black"
            contributors.setdefault(cell, set()).add(int(color))

    out = g.copy()
    for (r, c), colors in contributors.items():
        assert 1 <= len(colors) <= 2, "generator should avoid 3-way crossings"
        if len(colors) == 1:
            out[r, c] = next(iter(colors))
        else:
            out[r, c] = MIX_COLORS[frozenset(colors)]
    return out


def _sample_spaced(count: int, lo: int, hi: int, min_sep: int) -> list[int] | None:
    assert count >= 1 and min_sep >= 1
    span = hi - lo + 1
    extra = min_sep - 1
    compressed = span - extra * (count - 1)
    if compressed < count:
        return None
    picks = sorted(random.sample(range(compressed), count))
    return [lo + picks[i] + i * extra for i in range(count)]


def _transform_layout(
    height: int, width: int, seeds: dict[int, list[Coord]]
) -> tuple[int, int, dict[int, list[Coord]]]:
    transformed = {int(color): list(coords) for color, coords in seeds.items()}

    if random.random() < 0.5:
        transformed = {
            color: [(height - 1 - r, c) for r, c in coords]
            for color, coords in transformed.items()
        }
    if random.random() < 0.5:
        transformed = {
            color: [(r, width - 1 - c) for r, c in coords]
            for color, coords in transformed.items()
        }
    if random.random() < 0.5:
        transformed = {
            color: [(c, r) for r, c in coords]
            for color, coords in transformed.items()
        }
        height, width = width, height

    return height, width, transformed


def _render_input(height: int, width: int, seeds: dict[int, list[Coord]]) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    for color, coords in seeds.items():
        assert len(coords) == 2
        for r, c in coords:
            assert 0 <= r < height and 0 <= c < width
            assert int(grid[r, c]) == BLACK
            grid[r, c] = int(color)
    return grid


def _build_single_straight() -> tuple[int, int, dict[int, list[Coord]]] | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    color = random.choice(SEED_COLORS)
    horizontal = random.random() < 0.5

    if horizontal:
        row = random.randint(SEED_MARGIN, height - 1 - SEED_MARGIN)
        cols = _sample_spaced(2, SEED_MARGIN, width - 1 - SEED_MARGIN, MIN_SEGMENT_SPAN)
        if cols is None:
            return None
        seeds = {color: [(row, cols[0]), (row, cols[1])]}
    else:
        col = random.randint(SEED_MARGIN, width - 1 - SEED_MARGIN)
        rows = _sample_spaced(2, SEED_MARGIN, height - 1 - SEED_MARGIN, MIN_SEGMENT_SPAN)
        if rows is None:
            return None
        seeds = {color: [(rows[0], col), (rows[1], col)]}

    return _transform_layout(height, width, seeds)


def _build_straight_combo() -> tuple[int, int, dict[int, list[Coord]]] | None:
    pair_count = random.choices([2, 3], weights=[2, 3], k=1)[0]
    colors = random.sample(SEED_COLORS, k=pair_count)
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)

    if pair_count == 2:
        mode = random.choice(("cross", "parallel", "separate"))
        if mode == "cross":
            rows = _sample_spaced(3, SEED_MARGIN, height - 1 - SEED_MARGIN, 3)
            cols = _sample_spaced(3, SEED_MARGIN, width - 1 - SEED_MARGIN, 3)
            if rows is None or cols is None:
                return None
            seeds = {
                colors[0]: [(rows[1], cols[0]), (rows[1], cols[2])],
                colors[1]: [(rows[0], cols[1]), (rows[2], cols[1])],
            }
            return _transform_layout(height, width, seeds)

        if mode == "parallel":
            same_orientation = random.random() < 0.5
            if same_orientation:
                horizontal = random.random() < 0.5
                if horizontal:
                    rows = _sample_spaced(2, SEED_MARGIN, height - 1 - SEED_MARGIN, 3)
                    cols = _sample_spaced(4, SEED_MARGIN, width - 1 - SEED_MARGIN, 2)
                    if rows is None or cols is None:
                        return None
                    seeds = {
                        colors[0]: [(rows[0], cols[0]), (rows[0], cols[1])],
                        colors[1]: [(rows[1], cols[2]), (rows[1], cols[3])],
                    }
                else:
                    cols = _sample_spaced(2, SEED_MARGIN, width - 1 - SEED_MARGIN, 3)
                    rows = _sample_spaced(4, SEED_MARGIN, height - 1 - SEED_MARGIN, 2)
                    if rows is None or cols is None:
                        return None
                    seeds = {
                        colors[0]: [(rows[0], cols[0]), (rows[1], cols[0])],
                        colors[1]: [(rows[2], cols[1]), (rows[3], cols[1])],
                    }
                return _transform_layout(height, width, seeds)

        rows = _sample_spaced(3, SEED_MARGIN, height - 1 - SEED_MARGIN, 3)
        cols = _sample_spaced(5, SEED_MARGIN, width - 1 - SEED_MARGIN, 2)
        if rows is None or cols is None:
            return None
        seeds = {
            colors[0]: [(rows[0], cols[0]), (rows[0], cols[1])],
            colors[1]: [(rows[1], cols[3]), (rows[2], cols[3])],
        }
        return _transform_layout(height, width, seeds)

    mode = random.choice(("vertical_spine", "horizontal_spine", "parallel3"))
    if mode == "vertical_spine":
        rows = _sample_spaced(4, SEED_MARGIN, height - 1 - SEED_MARGIN, 2)
        cols = _sample_spaced(3, SEED_MARGIN, width - 1 - SEED_MARGIN, 3)
        if rows is None or cols is None:
            return None
        seeds = {
            colors[0]: [(rows[0], cols[1]), (rows[3], cols[1])],
            colors[1]: [(rows[1], cols[0]), (rows[1], cols[2])],
            colors[2]: [(rows[2], cols[0]), (rows[2], cols[2])],
        }
    elif mode == "horizontal_spine":
        rows = _sample_spaced(3, SEED_MARGIN, height - 1 - SEED_MARGIN, 3)
        cols = _sample_spaced(4, SEED_MARGIN, width - 1 - SEED_MARGIN, 2)
        if rows is None or cols is None:
            return None
        seeds = {
            colors[0]: [(rows[1], cols[0]), (rows[1], cols[3])],
            colors[1]: [(rows[0], cols[1]), (rows[2], cols[1])],
            colors[2]: [(rows[0], cols[2]), (rows[2], cols[2])],
        }
    else:
        horizontal = random.random() < 0.5
        if horizontal:
            rows = _sample_spaced(3, SEED_MARGIN, height - 1 - SEED_MARGIN, 2)
            cols = _sample_spaced(6, SEED_MARGIN, width - 1 - SEED_MARGIN, 1)
            if rows is None or cols is None:
                return None
            seeds = {
                colors[0]: [(rows[0], cols[0]), (rows[0], cols[1])],
                colors[1]: [(rows[1], cols[2]), (rows[1], cols[3])],
                colors[2]: [(rows[2], cols[4]), (rows[2], cols[5])],
            }
        else:
            cols = _sample_spaced(3, SEED_MARGIN, width - 1 - SEED_MARGIN, 2)
            rows = _sample_spaced(6, SEED_MARGIN, height - 1 - SEED_MARGIN, 1)
            if rows is None or cols is None:
                return None
            seeds = {
                colors[0]: [(rows[0], cols[0]), (rows[1], cols[0])],
                colors[1]: [(rows[2], cols[1]), (rows[3], cols[1])],
                colors[2]: [(rows[4], cols[2]), (rows[5], cols[2])],
            }

    return _transform_layout(height, width, seeds)


def _build_blocked_l() -> tuple[int, int, dict[int, list[Coord]]] | None:
    pair_count = random.choice((2, 3))
    colors = random.sample(SEED_COLORS, k=pair_count)
    height = random.randint(max(GRID_MIN, 12), GRID_MAX)
    width = random.randint(max(GRID_MIN, 12), GRID_MAX)

    rows = _sample_spaced(3, SEED_MARGIN, height - 1 - SEED_MARGIN, 4)
    cols = _sample_spaced(4, SEED_MARGIN, width - 1 - SEED_MARGIN, 2)
    if rows is None or cols is None:
        return None

    r1, r2, r3 = rows
    c0, c1, c2, c3 = cols
    seeds: dict[int, list[Coord]] = {
        colors[0]: [(r1, c1), (r2, c2)],
        colors[1]: [(r2, c1), (r3, c1)],
    }

    if pair_count == 3:
        mode = random.choice(("cross_l", "cross_blocker"))
        if mode == "cross_l":
            mid_rows = list(range(r1 + 1, r2))
            if not mid_rows:
                return None
            r_mid = random.choice(mid_rows)
            seeds[colors[2]] = [(r_mid, c0), (r_mid, c3)]
        else:
            mid_rows = list(range(r2 + 1, r3))
            if not mid_rows:
                return None
            r_mid = random.choice(mid_rows)
            seeds[colors[2]] = [(r_mid, c0), (r_mid, c3)]

    return _transform_layout(height, width, seeds)


def _is_visually_clear(inp: np.ndarray, out: np.ndarray) -> bool:
    path_pixels = int(np.count_nonzero((out != BLACK) & (inp == BLACK)))
    if path_pixels < 2:
        return False
    if np.count_nonzero(inp) > (inp.shape[0] * inp.shape[1]) // 6:
        return False
    return True


def generate_input() -> np.ndarray:
    builders = (
        _build_single_straight,
        _build_straight_combo,
        _build_blocked_l,
    )

    for _ in range(GEN_RETRIES):
        builder = random.choices(builders, weights=[1, 3, 3], k=1)[0]
        built = builder()
        if built is None:
            continue
        height, width, seeds = built
        try:
            inp = _render_input(height, width, seeds)
            out = main(inp)
        except AssertionError:
            continue
        if not _is_visually_clear(inp, out):
            continue
        return inp
    raise RuntimeError("failed to generate a valid ARC input")
