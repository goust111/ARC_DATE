import random
from collections import Counter, deque

import numpy as np


BG = 0
PALETTE_ORDER = (1, 2, 3, 4, 6, 7, 8, 9)
ORDER_RANK = {color: idx for idx, color in enumerate(PALETTE_ORDER)}

DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

QUADRANTS = {
    "NW": (DIRS["R"], DIRS["D"]),
    "NE": (DIRS["L"], DIRS["D"]),
    "SE": (DIRS["L"], DIRS["U"]),
    "SW": (DIRS["R"], DIRS["U"]),
}

ARROW_PATTERNS = {
    frozenset({(0, 1), (1, 0), (1, 1), (1, 2)}): {"tip": (0, 1), "dir": DIRS["U"]},
    frozenset({(0, 0), (0, 1), (0, 2), (1, 1)}): {"tip": (1, 1), "dir": DIRS["D"]},
    frozenset({(0, 1), (1, 0), (1, 1), (2, 1)}): {"tip": (1, 0), "dir": DIRS["L"]},
    frozenset({(0, 0), (1, 0), (1, 1), (2, 0)}): {"tip": (1, 1), "dir": DIRS["R"]},
}


def _add(pos: tuple[int, int], delta: tuple[int, int], k: int = 1) -> tuple[int, int]:
    return pos[0] + delta[0] * k, pos[1] + delta[1] * k


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


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


def _color_order_key(color: int) -> tuple[int, int]:
    return (ORDER_RANK.get(color, len(PALETTE_ORDER)), color)


def _perp(direction: tuple[int, int]) -> tuple[int, int]:
    if direction[0] != 0:
        return 0, 1
    return 1, 0


def _arrow_cells_from_tip(
    tip: tuple[int, int], direction: tuple[int, int]
) -> set[tuple[int, int]]:
    back = _add(tip, direction, -1)
    pr, pc = _perp(direction)
    return {
        tip,
        back,
        (back[0] + pr, back[1] + pc),
        (back[0] - pr, back[1] - pc),
    }


def _segment_cells(
    start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("segment must be orthogonal")


def _find_components_by_color(grid: np.ndarray) -> dict[int, list[list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    by_color: dict[int, list[list[tuple[int, int]]]] = {}
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        _in_bounds((h, w), nr, nc)
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            by_color.setdefault(color, []).append(coords)
    return by_color


def _detect_arrow(coords: list[tuple[int, int]]) -> dict[str, tuple[int, int]]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0 = min(rows), min(cols)
    norm = frozenset((r - r0, c - c0) for r, c in coords)
    assert norm in ARROW_PATTERNS, f"unrecognized arrow shape: {sorted(norm)}"
    spec = ARROW_PATTERNS[norm]
    tip = (r0 + spec["tip"][0], c0 + spec["tip"][1])
    return {"tip": tip, "dir": spec["dir"]}


def _ray_cells(
    grid: np.ndarray, start: tuple[int, int], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = grid.shape
    cells: list[tuple[int, int]] = []
    r, c = start
    while _in_bounds((h, w), r, c) and int(grid[r, c]) == BG:
        cells.append((r, c))
        r, c = _add((r, c), direction)
    return cells


def _solve_pair(
    grid: np.ndarray, components: list[list[tuple[int, int]]]
) -> tuple[list[dict[str, tuple[int, int]]], tuple[int, int]]:
    infos = [_detect_arrow(comp) for comp in components]
    dirs = [info["dir"] for info in infos]
    assert dirs[0][0] != dirs[1][0] or dirs[0][1] != dirs[1][1], "duplicate directions"
    assert dirs[0][0] != dirs[1][0] and dirs[0][1] != dirs[1][1], "expected one horizontal and one vertical arrow"

    ray_sets = []
    for info in infos:
        launch = _add(info["tip"], info["dir"])
        ray_sets.append(set(_ray_cells(grid, launch, info["dir"])))

    common = ray_sets[0] & ray_sets[1]
    assert len(common) == 1, f"expected unique bend, got {sorted(common)}"
    corner = next(iter(common))
    return infos, corner


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored arrow pair by an orthogonal L path whose bend is fixed by the two arrow directions.

    Input structure:
    - The background is black.
    - Every non-black component is a small monochrome arrowhead made of four pixels in a `T` shape.
    - Each non-black color appears in exactly two separate arrowheads, and the two arrows of one color point along one horizontal and one vertical axis.

    Rule:
    - Pair the two arrowheads of each color.
    - For each arrowhead, identify its tip cell and extend a straight forward ray from the cell immediately in front of that tip.
    - The two forward rays of a same-colored pair meet at one background cell; that meeting cell is the unique bend of the path.
    - Paint both ray segments from the two arrows to that bend in the pair color, but never overwrite the original arrowhead pixels.
    - If paths of different colors cross, paint color pairs in the fixed ARC palette order, so later colors overwrite earlier path pixels at the crossing.

    Key insight:
    - The local `T` shape reveals an arrow direction, and the final bend is not guessed from proximity: it is the one cell simultaneously reached by the two forward rays launched from the paired arrows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    by_color = _find_components_by_color(g)
    out = g.copy()
    original_fg = g != BG

    for color in sorted(by_color, key=_color_order_key):
        components = by_color[color]
        assert len(components) == 2, f"expected exactly two objects for color {color}"
        infos, corner = _solve_pair(g, components)
        for info in infos:
            launch = _add(info["tip"], info["dir"])
            for r, c in _segment_cells(launch, corner):
                if not original_fg[r, c]:
                    out[r, c] = color

    return out


GRID_MIN = 17
GRID_MAX = 28
PAIR_COUNT_OPTIONS = (2, 3)
LEG_LEN_MIN = 2
LEG_LEN_MAX = 8
MAX_RETRIES = 300
PAIR_PLACEMENT_RETRIES = 120
USE_CROSSING_PROB = 0.5


def _pad8(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    padded = set(cells)
    for r, c in cells:
        for nr, nc in _neighbors8(r, c):
            padded.add((nr, nc))
    return padded


def _all_in_bounds(shape: tuple[int, int], cells: set[tuple[int, int]]) -> bool:
    return all(_in_bounds(shape, r, c) for r, c in cells)


def _valid_lengths(
    shape: tuple[int, int], corner: tuple[int, int], direction: tuple[int, int]
) -> list[int]:
    valid = []
    for length in range(LEG_LEN_MIN, LEG_LEN_MAX + 1):
        tip = _add(corner, direction, -length)
        cells = _arrow_cells_from_tip(tip, direction)
        if _all_in_bounds(shape, cells):
            valid.append(length)
    return valid


def _build_pair(
    color: int,
    corner: tuple[int, int],
    quadrant: str,
    len_h: int,
    len_v: int,
) -> dict[str, object]:
    dir_h, dir_v = QUADRANTS[quadrant]
    tip_h = _add(corner, dir_h, -len_h)
    tip_v = _add(corner, dir_v, -len_v)
    comp_h = _arrow_cells_from_tip(tip_h, dir_h)
    comp_v = _arrow_cells_from_tip(tip_v, dir_v)
    path_h = {_add(tip_h, dir_h, k) for k in range(1, len_h + 1)}
    path_v = {_add(tip_v, dir_v, k) for k in range(1, len_v + 1)}
    return {
        "color": color,
        "corner": corner,
        "quadrant": quadrant,
        "components": [comp_h, comp_v],
        "object_cells": comp_h | comp_v,
        "path_cells": path_h | path_v,
    }


def _pair_self_consistent(pair: dict[str, object]) -> bool:
    comps = pair["components"]
    if comps[0] & _pad8(comps[1]):
        return False
    if pair["object_cells"] & pair["path_cells"]:
        return False
    return True


def _pair_fits(
    pair: dict[str, object],
    shape: tuple[int, int],
    existing_objects: set[tuple[int, int]],
    existing_paths: set[tuple[int, int]],
    allowed_overlaps: set[tuple[int, int]],
) -> bool:
    object_cells = pair["object_cells"]
    path_cells = pair["path_cells"]
    if not _all_in_bounds(shape, object_cells | path_cells):
        return False
    if not _pair_self_consistent(pair):
        return False

    padded_existing_objects = _pad8(existing_objects)
    if object_cells & padded_existing_objects:
        return False
    if object_cells & existing_paths:
        return False

    if any(cell in padded_existing_objects for cell in path_cells):
        return False

    overlaps = path_cells & existing_paths
    if not overlaps.issubset(allowed_overlaps):
        return False
    return True


def _random_corner(shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    return random.randint(3, h - 4), random.randint(3, w - 4)


def _try_place_pair(
    shape: tuple[int, int],
    color: int,
    quadrant: str,
    existing_objects: set[tuple[int, int]],
    existing_paths: set[tuple[int, int]],
    *,
    allowed_overlaps: set[tuple[int, int]] | None = None,
    forced_corner: tuple[int, int] | None = None,
) -> dict[str, object] | None:
    allowed = set() if allowed_overlaps is None else set(allowed_overlaps)
    for _ in range(PAIR_PLACEMENT_RETRIES):
        corner = forced_corner if forced_corner is not None else _random_corner(shape)
        dir_h, dir_v = QUADRANTS[quadrant]
        valid_h = _valid_lengths(shape, corner, dir_h)
        valid_v = _valid_lengths(shape, corner, dir_v)
        if not valid_h or not valid_v:
            if forced_corner is not None:
                return None
            continue
        len_h = random.choice(valid_h)
        len_v = random.choice(valid_v)
        pair = _build_pair(color, corner, quadrant, len_h, len_v)
        if _pair_fits(pair, shape, existing_objects, existing_paths, allowed):
            return pair
        if forced_corner is not None:
            return None
    return None


def _render_input(shape: tuple[int, int], pairs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for pair in pairs:
        color = int(pair["color"])
        for r, c in pair["object_cells"]:
            grid[r, c] = color
    return grid


def _render_output(inp: np.ndarray, pairs: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    original_fg = inp != BG
    by_color = {int(pair["color"]): pair for pair in pairs}
    for color in sorted(by_color, key=_color_order_key):
        for r, c in by_color[color]["path_cells"]:
            if not original_fg[r, c]:
                out[r, c] = color
    return out


def _path_overlap_count(pairs: list[dict[str, object]]) -> int:
    counts = Counter()
    for pair in pairs:
        counts.update(pair["path_cells"])
    return max(counts.values(), default=0)


def _cross_quadrants() -> tuple[str, str]:
    return random.choice((("NW", "SE"), ("NE", "SW")))


def _gen_attempt(
    shape: tuple[int, int],
    colors: list[int],
    quadrants: list[str],
    use_crossing: bool,
) -> np.ndarray | None:
    pairs: list[dict[str, object]] = []
    existing_objects: set[tuple[int, int]] = set()
    existing_paths: set[tuple[int, int]] = set()

    if use_crossing:
        cross_a, cross_b = quadrants[0], quadrants[1]
        for _ in range(PAIR_PLACEMENT_RETRIES):
            shared_corner = _random_corner(shape)
            pair_a = _try_place_pair(
                shape,
                colors[0],
                cross_a,
                existing_objects,
                existing_paths,
                forced_corner=shared_corner,
            )
            if pair_a is None:
                continue
            pair_b = _try_place_pair(
                shape,
                colors[1],
                cross_b,
                pair_a["object_cells"],
                pair_a["path_cells"],
                allowed_overlaps={shared_corner},
                forced_corner=shared_corner,
            )
            if pair_b is None:
                continue
            if (pair_a["path_cells"] & pair_b["path_cells"]) != {shared_corner}:
                continue
            pairs.extend([pair_a, pair_b])
            existing_objects |= pair_a["object_cells"] | pair_b["object_cells"]
            existing_paths |= pair_a["path_cells"] | pair_b["path_cells"]
            break
        if len(pairs) < 2:
            return None
        start_idx = 2
    else:
        start_idx = 0

    for idx in range(start_idx, len(colors)):
        pair = _try_place_pair(
            shape,
            colors[idx],
            quadrants[idx],
            existing_objects,
            existing_paths,
        )
        if pair is None:
            return None
        pairs.append(pair)
        existing_objects |= pair["object_cells"]
        existing_paths |= pair["path_cells"]

    inp = _render_input(shape, pairs)
    expected = _render_output(inp, pairs)

    if use_crossing and _path_overlap_count(pairs) < 2:
        return None
    if not use_crossing and _path_overlap_count(pairs) != 1:
        return None
    if int(np.argmax(np.bincount(inp.ravel(), minlength=10))) != BG:
        return None
    if sorted(int(v) for v in np.unique(inp) if int(v) != BG) != sorted(colors):
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    shape = (
        random.randint(GRID_MIN, GRID_MAX),
        random.randint(GRID_MIN, GRID_MAX),
    )
    pair_count = random.choice(PAIR_COUNT_OPTIONS)
    use_crossing = bool(random.random() < USE_CROSSING_PROB)
    if use_crossing:
        pair_count = max(pair_count, 2)
    colors = random.sample(PALETTE_ORDER, pair_count)

    quadrants: list[str]
    if use_crossing:
        qa, qb = _cross_quadrants()
        quadrants = [qa, qb]
        quadrants.extend(random.choice(tuple(QUADRANTS)) for _ in range(pair_count - 2))
    else:
        quadrants = [random.choice(tuple(QUADRANTS)) for _ in range(pair_count)]

    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(shape, colors, quadrants, use_crossing)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid arrow-pair input")
