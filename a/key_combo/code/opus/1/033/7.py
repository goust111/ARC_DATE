import random
from collections import Counter

import numpy as np


PATTERN_COLOR_POOL = (1, 3, 8, 9)
OCCLUDER_COLOR_POOL = (2, 4, 5, 6, 7)
PERIOD_VALUES = (2, 3)
GRID_MIN = 14
GRID_MAX = 20
OBJECT_COUNT_VALUES = (4, 5)
GEN_ATTEMPTS = 600

NON_SQUARE_CHOICES = (
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
    (3, 4),
    (4, 3),
    (2, 5),
    (5, 2),
)
EVEN_SQUARE_SIDES = (2, 2, 4)
ODD_SQUARE_SIDES = (3, 3, 5)
EXTRA_BRANCH_CHOICES = ("rect", "even", "odd")

_RNG = random.Random()
_PERIOD_BAG: list[int] = []
_COLOR_COUNT_BAG: list[int] = []
_OBJECT_COUNT_BAG: list[int] = []


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover periodic colors beneath square occluders while leaving non-square blockers unchanged.

    Input structure:
    - The grid is a 2D color pattern formed by repeating one hidden 2x2 or 3x3 tile.
    - Four or five non-overlapping monochrome solid rectangles cover parts of that periodic pattern.
    - Rectangle colors are distinct from the periodic tile colors, and the set of rectangles includes
      at least one non-square, one even-sided square, and one odd-sided square.

    Rule:
    - Infer the unique smallest square tile period, using only the visible periodic cells outside the rectangles.
    - Compare the input with that recovered periodic background to locate the monochrome rectangular occluders.
    - Leave every non-square rectangle unchanged.
    - Replace every square rectangle by the recovered periodic colors underneath it; if the square side is odd,
      repaint its single center cell with that rectangle's own original color.

    Key insight:
    - The hidden periodic layer determines what every square is hiding, but geometry decides what happens next:
      non-squares stay opaque, even squares disappear completely, and odd squares disappear except for a center mark
      in their original occluder color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    candidates = []
    for period in PERIOD_VALUES:
        candidate = _period_candidate(g, period)
        if candidate is not None:
            candidates.append(candidate)

    assert candidates, "expected at least one valid periodic reconstruction"
    best_period = min(candidate["period"] for candidate in candidates)
    best = [candidate for candidate in candidates if candidate["period"] == best_period]
    assert len(best) == 1, "visible cells must determine one smallest periodic tile"
    solved = best[0]

    out = g.copy()
    for rect in solved["rectangles"]:
        if rect["h"] != rect["w"]:
            continue
        r0 = rect["r"]
        c0 = rect["c"]
        side = rect["h"]
        out[r0 : r0 + side, c0 : c0 + side] = solved["base"][r0 : r0 + side, c0 : c0 + side]
        if side % 2 == 1:
            out[r0 + side // 2, c0 + side // 2] = rect["color"]

    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case["input"]
    raise RuntimeError("failed to generate a valid periodic-occluder input")


def _period_candidate(grid: np.ndarray, period: int) -> dict | None:
    tile = _majority_tile(grid, period)
    if tile is None:
        return None
    if _minimal_square_period(tile) != period:
        return None
    base = _tile_to_grid(tile, grid.shape)
    mismatch = grid != base
    rectangles = _extract_rectangles(grid, mismatch)
    if rectangles is None:
        return None
    return {
        "period": period,
        "tile": tile,
        "base": base,
        "rectangles": rectangles,
    }


def _majority_tile(grid: np.ndarray, period: int) -> np.ndarray | None:
    h, w = grid.shape
    tile = np.zeros((period, period), dtype=int)

    for pr in range(period):
        for pc in range(period):
            values = [int(grid[r, c]) for r in range(pr, h, period) for c in range(pc, w, period)]
            if not values:
                return None
            counts = Counter(values).most_common()
            if len(counts) > 1 and counts[0][1] == counts[1][1]:
                return None
            tile[pr, pc] = counts[0][0]

    colors = {int(value) for value in tile.flat}
    if not 2 <= len(colors) <= 3:
        return None
    if not _tile_uses_both_axes(tile):
        return None
    return tile


def _minimal_square_period(tile: np.ndarray) -> int:
    p = tile.shape[0]
    assert tile.shape == (p, p)
    for candidate in PERIOD_VALUES:
        if candidate > p:
            continue
        ok = True
        for r in range(p):
            for c in range(p):
                if int(tile[r, c]) != int(tile[r % candidate, c % candidate]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return candidate
    return p


def _tile_uses_both_axes(tile: np.ndarray) -> bool:
    row_patterns = {tuple(map(int, row)) for row in tile}
    col_patterns = {tuple(map(int, tile[:, c])) for c in range(tile.shape[1])}
    if len(row_patterns) < 2 or len(col_patterns) < 2:
        return False
    if not any(len(set(map(int, row))) > 1 for row in tile):
        return False
    if not any(len(set(map(int, tile[:, c]))) > 1 for c in range(tile.shape[1])):
        return False
    return True


def _tile_to_grid(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    p = tile.shape[0]
    rows = np.arange(h)[:, None] % p
    cols = np.arange(w)[None, :] % p
    return tile[rows, cols]


def _extract_rectangles(grid: np.ndarray, mismatch: np.ndarray) -> list[dict] | None:
    components = _components(mismatch)
    if not 4 <= len(components) <= 6:
        return None

    rectangles = []
    seen_colors: set[int] = set()
    has_rect = False
    has_even_square = False
    has_odd_square = False

    for coords in components:
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        area = h * w
        if len(coords) != area:
            return None

        color = int(grid[r0, c0])
        if color in seen_colors:
            return None
        seen_colors.add(color)

        block = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(block == color):
            return None
        if int(np.count_nonzero(grid == color)) != area:
            return None

        if h == w:
            if h % 2 == 0:
                has_even_square = True
            else:
                has_odd_square = True
        else:
            has_rect = True

        rectangles.append(
            {
                "r": r0,
                "c": c0,
                "h": h,
                "w": w,
                "color": color,
            }
        )

    if not (has_rect and has_even_square and has_odd_square):
        return None
    rectangles.sort(key=lambda rect: (rect["r"], rect["c"], rect["h"], rect["w"]))
    return rectangles


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _generate_case() -> dict | None:
    height = _RNG.randint(GRID_MIN, GRID_MAX)
    width = _RNG.randint(GRID_MIN, GRID_MAX)
    period = _draw_period()
    color_count = _draw_color_count()
    tile = _sample_tile(period, color_count)
    if tile is None:
        return None

    base = _tile_to_grid(tile, (height, width))
    object_count = _draw_object_count()
    specs = _sample_occluder_specs(object_count)
    occluders = _place_occluders(height, width, specs)
    if occluders is None:
        return None

    covered = np.zeros((height, width), dtype=bool)
    grid = base.copy()
    for rect in occluders:
        r0 = rect["r"]
        c0 = rect["c"]
        h = rect["h"]
        w = rect["w"]
        covered[r0 : r0 + h, c0 : c0 + w] = True
        grid[r0 : r0 + h, c0 : c0 + w] = rect["color"]

    if not _visible_period_unique(base, covered, period):
        return None
    if not _phase_majorities_are_strict(grid, base, period):
        return None

    expected = _oracle_output(grid, base, occluders)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return {
        "input": grid,
        "base": base,
        "tile": tile,
        "period": period,
        "occluders": occluders,
        "expected": expected,
    }


def _draw_period() -> int:
    if not _PERIOD_BAG:
        _PERIOD_BAG.extend([2, 2, 3, 3])
        _RNG.shuffle(_PERIOD_BAG)
    return _PERIOD_BAG.pop()


def _draw_color_count() -> int:
    if not _COLOR_COUNT_BAG:
        _COLOR_COUNT_BAG.extend([2, 2, 3, 3])
        _RNG.shuffle(_COLOR_COUNT_BAG)
    return _COLOR_COUNT_BAG.pop()


def _draw_object_count() -> int:
    if not _OBJECT_COUNT_BAG:
        _OBJECT_COUNT_BAG.extend([4, 4, 5, 5])
        _RNG.shuffle(_OBJECT_COUNT_BAG)
    return _OBJECT_COUNT_BAG.pop()


def _sample_tile(period: int, color_count: int) -> np.ndarray | None:
    colors = _RNG.sample(PATTERN_COLOR_POOL, color_count)
    for _ in range(250):
        tile = np.array([_RNG.choices(colors, k=period) for _ in range(period)], dtype=int)
        if {int(value) for value in tile.flat} != set(colors):
            continue
        if _minimal_square_period(tile) != period:
            continue
        if not _tile_uses_both_axes(tile):
            continue
        return tile
    return None


def _sample_occluder_specs(object_count: int) -> list[dict]:
    categories = ["rect", "even", "odd"]
    while len(categories) < object_count:
        categories.append(_RNG.choice(EXTRA_BRANCH_CHOICES))
    _RNG.shuffle(categories)

    colors = _RNG.sample(OCCLUDER_COLOR_POOL, object_count)
    specs: list[dict] = []
    for category, color in zip(categories, colors):
        if category == "rect":
            h, w = _RNG.choice(NON_SQUARE_CHOICES)
        elif category == "even":
            side = _RNG.choice(EVEN_SQUARE_SIDES)
            h = side
            w = side
        else:
            side = _RNG.choice(ODD_SQUARE_SIDES)
            h = side
            w = side
        specs.append({"kind": category, "h": h, "w": w, "color": color})

    specs.sort(key=lambda spec: (-(spec["h"] * spec["w"]), spec["kind"]))
    return specs


def _place_occluders(height: int, width: int, specs: list[dict]) -> list[dict] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placed: list[dict] = []

    total_area = sum(spec["h"] * spec["w"] for spec in specs)
    if total_area > int(height * width * 0.38):
        return None

    for spec in specs:
        positions = _free_positions(occupied, spec["h"], spec["w"])
        if not positions:
            return None
        r0, c0 = _RNG.choice(positions)
        occupied[r0 : r0 + spec["h"], c0 : c0 + spec["w"]] = True
        placed.append(
            {
                "r": r0,
                "c": c0,
                "h": spec["h"],
                "w": spec["w"],
                "color": spec["color"],
                "kind": spec["kind"],
            }
        )

    placed.sort(key=lambda rect: (rect["r"], rect["c"], rect["h"], rect["w"]))
    return placed


def _free_positions(occupied: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    height, width = occupied.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(height - h + 1):
        rr0 = max(0, r0 - 1)
        rr1 = min(height, r0 + h + 1)
        for c0 in range(width - w + 1):
            cc0 = max(0, c0 - 1)
            cc1 = min(width, c0 + w + 1)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    return positions


def _visible_period_unique(base: np.ndarray, covered: np.ndarray, true_period: int) -> bool:
    valid_periods = []
    visible = ~covered
    for period in PERIOD_VALUES:
        phase_map: dict[tuple[int, int], int] = {}
        ok = True
        for r, c in zip(*np.where(visible)):
            key = (int(r) % period, int(c) % period)
            color = int(base[r, c])
            prev = phase_map.get(key)
            if prev is None:
                phase_map[key] = color
            elif prev != color:
                ok = False
                break
        if ok and len(phase_map) == period * period:
            valid_periods.append(period)
    return valid_periods == [true_period]


def _phase_majorities_are_strict(grid: np.ndarray, base: np.ndarray, period: int) -> bool:
    h, w = grid.shape
    for pr in range(period):
        for pc in range(period):
            values = [int(grid[r, c]) for r in range(pr, h, period) for c in range(pc, w, period)]
            counts = Counter(values)
            target = int(base[pr, pc])
            target_count = counts[target]
            best_other = max((count for color, count in counts.items() if color != target), default=0)
            if target_count <= best_other:
                return False
    return True


def _oracle_output(grid: np.ndarray, base: np.ndarray, occluders: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for rect in occluders:
        if rect["h"] != rect["w"]:
            continue
        r0 = rect["r"]
        c0 = rect["c"]
        side = rect["h"]
        out[r0 : r0 + side, c0 : c0 + side] = base[r0 : r0 + side, c0 : c0 + side]
        if side % 2 == 1:
            out[r0 + side // 2, c0 + side // 2] = rect["color"]
    return out
