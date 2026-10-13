from __future__ import annotations

from functools import lru_cache
from importlib.machinery import SourceFileLoader
from itertools import combinations, product

import numpy as np


_ARC = SourceFileLoader(
    "_shared_arc_utils",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC.Color
colors_present = _ARC.colors_present
find_components = _ARC.find_components
get_bg = _ARC.get_bg


BG = Color.BLACK
BAR_COLOR = Color.RED
SQUARE_COLOR = Color.TEAL
INPUT_COLORS = (Color.BLUE, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.MAROON)
DIRS8 = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)

GRID_MIN = 18
GRID_MAX = 28
SCATTER_DISTANCE_CHOICES = np.arange(4, 12)
MAX_GEN_RETRIES = 300
PLACE_RETRIES = 200


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _anchor_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    cr, cc = center
    return {(cr + dr, cc + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _anchor_rel_cells() -> set[tuple[int, int]]:
    return {(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _ray_direction(dr: int, dc: int) -> tuple[int, int] | None:
    if dr == 0 and dc == 0:
        return None
    if dr == 0:
        return 0, _sgn(dc)
    if dc == 0:
        return _sgn(dr), 0
    if abs(dr) == abs(dc):
        return _sgn(dr), _sgn(dc)
    return None


def _shift_cells(cells: set[tuple[int, int]], offset: tuple[int, int]) -> set[tuple[int, int]]:
    dr, dc = offset
    return {(r + dr, c + dc) for r, c in cells}


def _expand_cells(cells: set[tuple[int, int]], radius: int = 1) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                out.add((r + dr, c + dc))
    return out


def _compress_ray(center: tuple[int, int], direction: tuple[int, int], count: int) -> list[tuple[int, int]]:
    cr, cc = center
    dr, dc = direction
    return [(cr + (2 + idx) * dr, cc + (2 + idx) * dc) for idx in range(count)]


def _first_uncovered(cells_key: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return min(cells_key)


@lru_cache(maxsize=None)
def _decompose_key(cells_key: tuple[tuple[int, int], ...]) -> tuple[tuple[str, tuple[tuple[int, int], ...]], ...] | None:
    if not cells_key:
        return ()

    remaining = set(cells_key)
    r, c = _first_uncovered(cells_key)
    candidates: list[tuple[str, tuple[tuple[int, int], ...]]] = []

    square = tuple(sorted(((r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1))))
    if set(square) <= remaining:
        candidates.append(("square", square))

    hbar = tuple(sorted(((r, c), (r, c + 1), (r, c + 2))))
    if set(hbar) <= remaining:
        candidates.append(("bar", hbar))

    vbar = tuple(sorted(((r, c), (r + 1, c), (r + 2, c))))
    if set(vbar) <= remaining:
        candidates.append(("bar", vbar))

    for piece_type, piece_cells in candidates:
        nxt = tuple(sorted(remaining - set(piece_cells)))
        tail = _decompose_key(nxt)
        if tail is not None:
            return ((piece_type, piece_cells),) + tail
    return None


def _decompose_shape(cells: set[tuple[int, int]]) -> list[tuple[str, set[tuple[int, int]]]] | None:
    result = _decompose_key(tuple(sorted(cells)))
    if result is None:
        return None
    return [(piece_type, set(piece_cells)) for piece_type, piece_cells in result]


def _pattern_shape(pattern: tuple[tuple[tuple[int, int], int], ...]) -> set[tuple[int, int]]:
    cells = set(_anchor_rel_cells())
    for direction, count in pattern:
        for cell in _compress_ray((0, 0), direction, count):
            cells.add(cell)
    return cells


@lru_cache(maxsize=1)
def _valid_patterns() -> tuple[tuple[tuple[tuple[int, int], int], ...], ...]:
    patterns = []
    for ray_count in (1, 2, 3):
        for chosen_dirs in combinations(DIRS8, ray_count):
            for counts in product(range(1, 7), repeat=ray_count):
                if not (3 <= sum(counts) <= 6):
                    continue
                pattern = tuple(sorted(zip(chosen_dirs, counts)))
                shape = _pattern_shape(pattern)
                if _decompose_shape(shape) is not None:
                    patterns.append(pattern)
    patterns = sorted(set(patterns), key=lambda p: (len(p), sum(v for _, v in p), p))
    return tuple(patterns)


def _parse_color_object(grid: np.ndarray, color: int) -> tuple[tuple[int, int], dict[tuple[int, int], list[int]]]:
    comps = find_components(
        grid,
        bg=BG,
        connectivity=8,
        monochromatic=True,
        colors=[color],
        return_format="dict",
    )
    anchor_candidates = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if int(comp["area"]) == 9 and (r1 - r0 + 1, c1 - c0 + 1) == (3, 3):
            anchor_candidates.append(comp)
    assert len(anchor_candidates) == 1, f"color {color}: expected exactly one 3x3 anchor"

    anchor = anchor_candidates[0]
    r0, c0, r1, c1 = map(int, anchor["bbox"])
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), f"color {color}: anchor must be solid"
    center = (r0 + 1, c0 + 1)
    anchor_set = _anchor_cells(center)

    ray_groups: dict[tuple[int, int], list[int]] = {}
    all_cells = set(map(tuple, np.argwhere(grid == color)))
    for r, c in sorted(all_cells - anchor_set):
        dr = int(r - center[0])
        dc = int(c - center[1])
        direction = _ray_direction(dr, dc)
        assert direction is not None, f"color {color}: stray pixel must lie on a cardinal or diagonal ray"
        dist = max(abs(dr), abs(dc))
        assert dist >= 2, f"color {color}: stray pixel must lie outside the anchor"
        ray_groups.setdefault(direction, []).append(dist)

    assert 1 <= len(ray_groups) <= 3, f"color {color}: expected 1-3 rays"
    total = sum(len(v) for v in ray_groups.values())
    assert 3 <= total <= 6, f"color {color}: expected 3-6 stray pixels"
    for direction, dists in ray_groups.items():
        assert len(dists) == len(set(dists)), f"color {color}: duplicated positions on ray {direction}"
    return center, ray_groups


def _compressed_shape(center: tuple[int, int], ray_groups: dict[tuple[int, int], list[int]]) -> set[tuple[int, int]]:
    cells = set(_anchor_cells(center))
    for direction, dists in ray_groups.items():
        for cell in _compress_ray(center, direction, len(dists)):
            cells.add(cell)
    return cells


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Compress same-color ray pixels back to each 3x3 anchor, then decompose the compressed shape into squares and bars and recolor by primitive type.

    Input structure:
    - A black background contains 2-3 foreground colors.
    - Each foreground color has exactly one solid 3x3 anchor block plus 3-6 same-color stray pixels.
    - Every stray pixel lies on one of 1-3 cardinal or 45-degree rays that start at the anchor center and extend outside the block.

    Rule:
    - For each color independently, find its 3x3 anchor and group all same-color stray pixels by ray direction from the anchor center.
    - Compress every ray toward the anchor: a ray with k stray pixels becomes k consecutive cells starting from the nearest empty position just outside the anchor on that ray.
    - Treat the anchor plus the compressed pixels as one connected shape, then scan it in row-major order while preferring a 2x2 square whenever that choice still allows a complete decomposition; otherwise use a 1x3 or 3x1 bar.
    - Paint every square teal and every bar red on a black output canvas; clear the original scattered locations.

    Key insight:
    - The visible input shape is not what gets decomposed: the decomposition happens only after the ray pixels are geometrically compacted back against their anchor.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2
    bg = get_bg(grid)
    assert bg == BG

    out = np.full(grid.shape, BG, dtype=int)
    for color in colors_present(grid, exclude={bg}):
        center, ray_groups = _parse_color_object(grid, int(color))
        shape = _compressed_shape(center, ray_groups)
        decomp = _decompose_shape(shape)
        assert decomp is not None, f"color {color}: compressed shape is not decomposable"
        for piece_type, piece_cells in decomp:
            paint = SQUARE_COLOR if piece_type == "square" else BAR_COLOR
            for r, c in piece_cells:
                assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1]
                out[r, c] = paint
    return out


def _random_pattern(rng: np.random.Generator) -> tuple[tuple[tuple[int, int], int], ...]:
    patterns = _valid_patterns()
    weights = np.array([1 + len(pattern) for pattern in patterns], dtype=float)
    weights /= weights.sum()
    idx = int(rng.choice(len(patterns), p=weights))
    return patterns[idx]


def _random_distances(rng: np.random.Generator, count: int) -> list[int]:
    distances = sorted(int(v) for v in rng.choice(SCATTER_DISTANCE_CHOICES, size=count, replace=False))
    if count >= 2 and distances[-1] - distances[0] <= count:
        distances = sorted(int(v) for v in rng.choice(SCATTER_DISTANCE_CHOICES, size=count, replace=False))
    return distances


def _object_blueprint(
    pattern: tuple[tuple[tuple[int, int], int], ...],
    rng: np.random.Generator,
) -> dict[str, object]:
    scatter_rel = set()
    ray_distances: dict[tuple[int, int], list[int]] = {}
    for direction, count in pattern:
        dists = _random_distances(rng, count)
        ray_distances[direction] = dists
        dr, dc = direction
        for dist in dists:
            scatter_rel.add((dr * dist, dc * dist))

    anchor_rel = _anchor_rel_cells()
    compressed_rel = _pattern_shape(pattern)
    all_rel = anchor_rel | scatter_rel | compressed_rel
    padded_rel = _expand_cells(all_rel, radius=1)
    return {
        "pattern": pattern,
        "ray_distances": ray_distances,
        "anchor_rel": anchor_rel,
        "scatter_rel": scatter_rel,
        "compressed_rel": compressed_rel,
        "all_rel": all_rel,
        "padded_rel": padded_rel,
    }


def _choose_center(
    shape: tuple[int, int],
    blueprint: dict[str, object],
    reserved: set[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    h, w = shape
    padded_rel = blueprint["padded_rel"]
    min_r = -min(r for r, _ in padded_rel)
    max_r = (h - 1) - max(r for r, _ in padded_rel)
    min_c = -min(c for _, c in padded_rel)
    max_c = (w - 1) - max(c for _, c in padded_rel)
    if min_r > max_r or min_c > max_c:
        return None

    for _ in range(PLACE_RETRIES):
        center = (int(rng.integers(min_r, max_r + 1)), int(rng.integers(min_c, max_c + 1)))
        abs_padded = _shift_cells(padded_rel, center)
        if abs_padded.isdisjoint(reserved):
            return center
    return None


def _paint_object(
    grid: np.ndarray,
    center: tuple[int, int],
    color: int,
    blueprint: dict[str, object],
) -> None:
    for r, c in _shift_cells(blueprint["anchor_rel"], center):
        grid[r, c] = color
    for r, c in _shift_cells(blueprint["scatter_rel"], center):
        grid[r, c] = color


def _validate_instance(grid: np.ndarray) -> None:
    out = main(grid)
    assert isinstance(out, np.ndarray)
    assert out.shape == grid.shape
    assert set(np.unique(out)) <= {BG, BAR_COLOR, SQUARE_COLOR}
    assert np.count_nonzero(grid) == np.count_nonzero(out), "compression must preserve foreground area"
    assert not np.array_equal(grid, out), "task must not degenerate to identity"

    fg_colors = colors_present(grid, exclude={BG})
    assert 2 <= len(fg_colors) <= 3
    for color in fg_colors:
        center, ray_groups = _parse_color_object(grid, int(color))
        for dists in ray_groups.values():
            assert min(dists) >= 4, "generation should require visible compression"
        compressed = _compressed_shape(center, ray_groups)
        assert _decompose_shape(compressed) is not None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_RETRIES):
        h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
        w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
        grid = np.full((h, w), BG, dtype=int)
        reserved: set[tuple[int, int]] = set()

        color_count = int(rng.choice([2, 3], p=[0.45, 0.55]))
        colors = tuple(int(c) for c in rng.choice(INPUT_COLORS, size=color_count, replace=False))

        blueprints = []
        for _idx in range(color_count):
            pattern = _random_pattern(rng)
            blueprints.append(_object_blueprint(pattern, rng))

        placements = []
        ok = True
        for color, blueprint in zip(colors, blueprints):
            center = _choose_center(grid.shape, blueprint, reserved, rng)
            if center is None:
                ok = False
                break
            placements.append((color, center, blueprint))
            reserved |= _shift_cells(blueprint["padded_rel"], center)
        if not ok:
            continue

        for color, center, blueprint in placements:
            _paint_object(grid, center, color, blueprint)

        try:
            _validate_instance(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {MAX_GEN_RETRIES} attempts")
