from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

try:
    _ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
    if str(_ARC_UTILS_DIR) not in sys.path:
        sys.path.append(str(_ARC_UTILS_DIR))
    from arc_utils import Color, in_bounds
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        rows, cols = map(int, shape)
        return 0 <= int(r) < rows and 0 <= int(c) < cols


Coord = tuple[int, int]

GRID_SIZE = 15
GEN_CENTER = GRID_SIZE // 2
BACKGROUND = Color.BLACK
CROSS_COLOR = Color.BLUE
SEED_COLOR_POOL = [
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]
SEED_COUNT_CHOICES = [4, 5, 6, 7, 8]
SEED_COUNT_WEIGHTS = [4, 4, 3, 2, 1]
MAX_EVEN_SEEDS = 4
GEN_ATTEMPTS = 300
QUADRANTS = ("NW", "NE", "SW", "SE")


def _cross_center(grid: np.ndarray) -> Coord:
    rows = np.where(np.all(grid == CROSS_COLOR, axis=1))[0]
    cols = np.where(np.all(grid == CROSS_COLOR, axis=0))[0]
    assert len(rows) == 1 and len(cols) == 1, "expected one full blue row and one full blue column"
    return int(rows[0]), int(cols[0])


def _chebyshev(center: Coord, coord: Coord) -> int:
    cr, cc = center
    r, c = coord
    return max(abs(int(r) - int(cr)), abs(int(c) - int(cc)))


def _d4_orbit(coord: Coord, center: Coord, shape: tuple[int, int]) -> list[Coord]:
    r, c = coord
    cr, cc = center
    dr = int(r) - int(cr)
    dc = int(c) - int(cc)
    rels = [
        (dr, dc),
        (dc, -dr),
        (-dr, -dc),
        (-dc, dr),
        (dr, -dc),
        (-dc, -dr),
        (-dr, dc),
        (dc, dr),
    ]
    pts = {
        (int(cr + rr), int(cc + cc2))
        for rr, cc2 in rels
        if in_bounds(shape, int(cr + rr), int(cc + cc2))
    }
    return sorted(pts)


def _quadrant(coord: Coord, center: Coord) -> str:
    r, c = coord
    cr, cc = center
    if r < cr and c < cc:
        return "NW"
    if r < cr and c > cc:
        return "NE"
    if r > cr and c < cc:
        return "SW"
    if r > cr and c > cc:
        return "SE"
    raise ValueError(f"cross cell {coord} has no quadrant")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each off-cross seed through its D4 orbit and stamp only the orbit positions lying on even Chebyshev rings.

    Input structure:
    - The grid is an odd square with black background and one full blue cross made of a complete row and a complete column.
    - All non-blue off-cross foreground pixels are singleton seeds placed in the four quadrant regions.
    - Seeds may use any non-black, non-blue ARC color, and the other cells in their D4 orbits are background in the input.

    Rule:
    - Find the blue cross intersection and use it as the symmetry center.
    - For each off-cross seed, compute the deduplicated D4 orbit obtained by the square's axis reflections, diagonal reflections, and quarter turns around that center.
    - Keep the seed itself unchanged, but for every other orbit partner that is background in the input, stamp the seed color there iff that partner's Chebyshev distance from the center is even.
    - Never modify the blue cross; if multiple seeds claim one target, keep the closer seed.

    Key insight:
    - The symmetry orbit decides which positions are linked, while the even Chebyshev ring acts as a gate that allows only some orbit partners to receive the seed color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1] and g.shape[0] % 2 == 1, "expected an odd square grid"

    n = g.shape[0]
    center = _cross_center(g)
    assert center == (n // 2, n // 2), "expected the blue cross to pass through the grid center"

    out = g.copy()
    claims: dict[Coord, tuple[tuple[int, int, int], int]] = {}
    seeds: list[tuple[int, int, int]] = []
    for r in range(n):
        for c in range(n):
            color = int(g[r, c])
            if r == center[0] or c == center[1]:
                continue
            if color not in (BACKGROUND, CROSS_COLOR):
                seeds.append((r, c, color))

    assert seeds, "expected at least one off-cross seed"

    for r, c, color in seeds:
        priority = (_chebyshev(center, (r, c)), int(r), int(c))
        for rr, cc in _d4_orbit((r, c), center, g.shape):
            if (rr, cc) == (r, c):
                continue
            if rr == center[0] or cc == center[1]:
                continue
            if int(g[rr, cc]) != BACKGROUND:
                continue
            if _chebyshev(center, (rr, cc)) % 2 != 0:
                continue
            prev = claims.get((rr, cc))
            if prev is None or priority < prev[0]:
                claims[(rr, cc)] = (priority, color)

    for (r, c), (_priority, color) in claims.items():
        out[r, c] = int(color)

    return out


def _canonical_pairs(radius: int) -> list[tuple[int, int]]:
    return [(a, b) for b in range(1, radius + 1) for a in range(1, b + 1)]


def _orbit_from_pair(pair: tuple[int, int]) -> list[Coord]:
    a, b = pair
    return _d4_orbit((GEN_CENTER + a, GEN_CENTER + b), (GEN_CENTER, GEN_CENTER), (GRID_SIZE, GRID_SIZE))


def _seed_positions_for_quadrant(pair: tuple[int, int], quadrant: str) -> list[Coord]:
    orbit = _orbit_from_pair(pair)
    return [coord for coord in orbit if _quadrant(coord, (GEN_CENTER, GEN_CENTER)) == quadrant]


def _well_spaced(coords: list[Coord]) -> bool:
    for i, (r0, c0) in enumerate(coords):
        for r1, c1 in coords[i + 1 :]:
            if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                return False
    return True


def _quadrant_targets(seed_count: int) -> list[str]:
    targets: list[str] = []
    while len(targets) < seed_count:
        block = list(QUADRANTS)
        random.shuffle(block)
        targets.extend(block)
    return targets[:seed_count]


def _sample_pairs(seed_count: int) -> list[tuple[int, int]]:
    pairs = _canonical_pairs(GEN_CENTER)
    even_pairs = [pair for pair in pairs if pair[1] % 2 == 0]
    odd_pairs = [pair for pair in pairs if pair[1] % 2 == 1]

    even_count = random.randint(1, min(MAX_EVEN_SEEDS, seed_count - 1))
    odd_count = seed_count - even_count

    selected = random.sample(even_pairs, even_count) + random.sample(odd_pairs, odd_count)
    random.shuffle(selected)
    return selected


def _validate_generated_example(inp: np.ndarray, seeds: list[tuple[int, int, int]]) -> bool:
    out = main(inp)
    added = int(((inp == BACKGROUND) & (out != BACKGROUND)).sum())
    if not 3 <= added <= 28:
        return False

    has_even = False
    has_odd = False
    center = (GEN_CENTER, GEN_CENTER)
    for r, c, color in seeds:
        orbit = _d4_orbit((r, c), center, inp.shape)
        if _chebyshev(center, (r, c)) % 2 == 0:
            has_even = True
        else:
            has_odd = True

        expected = {
            (rr, cc)
            for rr, cc in orbit
            if (rr, cc) != (r, c) and int(inp[rr, cc]) == BACKGROUND and _chebyshev(center, (rr, cc)) % 2 == 0
        }
        actual = {
            (rr, cc)
            for rr, cc in orbit
            if int(inp[rr, cc]) == BACKGROUND and int(out[rr, cc]) == int(color)
        }
        if actual != expected:
            return False

    return has_even and has_odd and np.array_equal(out[GEN_CENTER, :], inp[GEN_CENTER, :]) and np.array_equal(
        out[:, GEN_CENTER], inp[:, GEN_CENTER]
    )


def _gen_attempt() -> np.ndarray | None:
    grid = np.full((GRID_SIZE, GRID_SIZE), BACKGROUND, dtype=int)
    grid[GEN_CENTER, :] = CROSS_COLOR
    grid[:, GEN_CENTER] = CROSS_COLOR

    seed_count = random.choices(SEED_COUNT_CHOICES, weights=SEED_COUNT_WEIGHTS, k=1)[0]
    colors = random.sample(SEED_COLOR_POOL, seed_count)
    pairs = _sample_pairs(seed_count)
    targets = _quadrant_targets(seed_count)

    seeds: list[tuple[int, int, int]] = []
    seed_coords: list[Coord] = []
    for pair, color, quadrant in zip(pairs, colors, targets):
        options = _seed_positions_for_quadrant(pair, quadrant)
        assert options, f"orbit {pair} must reach quadrant {quadrant}"
        coord = random.choice(options)
        seed_coords.append(coord)
        seeds.append((coord[0], coord[1], int(color)))

    if not _well_spaced(seed_coords):
        return None

    for r, c, color in seeds:
        grid[r, c] = int(color)

    if not _validate_generated_example(grid, seeds):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
