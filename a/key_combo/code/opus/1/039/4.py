import random
from collections import deque

import numpy as np


BG = 0
LARGE_COLOR = 5
SMALL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

GRID_SIZES = (15, 17, 19, 21, 23)
LARGE_AREA_RANGE = (10, 20)
SMALL_COUNT_RANGE = (3, 5)
SMALL_AREA_RANGE = (2, 4)
NOISE_COUNT_RANGE = (4, 7)
MAX_GEN_ATTEMPTS = 600


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _round_half_up(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _bbox_center(coords: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    return _round_half_up((r0 + r1) / 2.0), _round_half_up((c0 + c1) / 2.0)


def _centroid(coords: np.ndarray) -> tuple[int, int]:
    return (
        _round_half_up(float(coords[:, 0].mean())),
        _round_half_up(float(coords[:, 1].mean())),
    )


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _find_components(grid: np.ndarray, bg: int | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g) if bg is None else int(bg)
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            arr = np.array(sorted(coords), dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": _bbox(arr),
                }
            )
    return comps


def _analyze(grid: np.ndarray) -> tuple[int, list[dict], dict, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    comps = _find_components(g, bg)
    solids = [comp for comp in comps if int(comp["area"]) > 1]
    assert solids, "expected at least one non-noise component"
    max_area = max(int(comp["area"]) for comp in solids)
    largest = [comp for comp in solids if int(comp["area"]) == max_area]
    assert len(largest) == 1, "expected a unique largest component"
    large = largest[0]
    small = [comp for comp in solids if comp is not large]
    return bg, comps, large, small


def _ring_output(
    shape: tuple[int, int],
    center: tuple[int, int],
    distance_to_color: dict[int, int],
    bg: int,
) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), int(bg), dtype=int)
    for r in range(h):
        for c in range(w):
            d = _chebyshev((r, c), center)
            if d in distance_to_color:
                out[r, c] = int(distance_to_color[d])
    return out


def _distance_map(small_components: list[dict], center: tuple[int, int]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for comp in small_components:
        coords = comp["coords"]
        d = min(_chebyshev((int(r), int(c)), center) for r, c in coords)
        assert d not in mapping, "small-component distances must be unique"
        mapping[d] = int(comp["color"])
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """Erase the unique largest component and use the smaller components to color concentric square rings around its centroid.

    Input structure:
    - The grid is an odd-sized square with black background and several 4-connected monochrome components.
    - Exactly one component is the unique largest non-noise object; the other meaningful components are smaller colored patches, plus isolated single-pixel noise.
    - The largest component is only a positional anchor, while each smaller component contributes one ring color.

    Rule:
    - Find all 4-connected monochrome components and discard area-1 noise.
    - Select the unique largest remaining component and compute its centroid by averaging row and column coordinates separately, rounding half up to integer cells.
    - Erase that largest component, then for each remaining component compute the minimum Chebyshev distance from any of its pixels to the centroid.
    - Fill the output grid with concentric square rings centered at the centroid: every cell whose Chebyshev distance matches one of those component distances takes that component's color, and all other cells stay background.

    Key insight:
    - The biggest component disappears completely; it only defines the center, while the smaller components encode a distance-to-color lookup that gets expanded into full square rings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, _comps, large, small = _analyze(g)
    center = _centroid(large["coords"])
    return _ring_output(g.shape, center, _distance_map(small, center), bg)


def _all_cells_on_distance(size: int, center: tuple[int, int], distance: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(size):
        for c in range(size):
            if _chebyshev((r, c), center) == distance:
                cells.append((r, c))
    return cells


def _is_clear_from_occupied(
    cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    size: int,
) -> bool:
    for r, c in cells:
        if not (0 <= r < size and 0 <= c < size):
            return False
        for dr, dc in NEIGHBORS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and (nr, nc) in occupied:
                return False
    return True


def _random_connected_offsets(area: int, max_extent: int = 3) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < int(area):
        frontier = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if max(abs(nr), abs(nc)) > max_extent or (nr, nc) in cells:
                    continue
                frontier.add((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(tuple(frontier)))
    return cells


def _generate_large_component(size: int, center: tuple[int, int]) -> set[tuple[int, int]] | None:
    area = random.randint(*LARGE_AREA_RANGE)
    for _ in range(300):
        offsets = _random_connected_offsets(area)
        if len(offsets) != area:
            continue
        cells = {(center[0] + dr, center[1] + dc) for dr, dc in offsets}
        if any(not (0 <= r < size and 0 <= c < size) for r, c in cells):
            continue
        coords = np.array(sorted(cells), dtype=int)
        if _centroid(coords) != center:
            continue
        if _bbox_center(coords) == center:
            continue
        height = int(coords[:, 0].max() - coords[:, 0].min() + 1)
        width = int(coords[:, 1].max() - coords[:, 1].min() + 1)
        if min(height, width) < 3:
            continue
        return cells
    return None


def _sample_center(size: int) -> tuple[int, int]:
    middle = size // 2
    candidates = [
        (r, c)
        for r in range(4, size - 4)
        for c in range(4, size - 4)
        if max(abs(r - middle), abs(c - middle)) >= 2
    ]
    return random.choice(candidates)


def _choose_distances(max_distance: int) -> tuple[list[int], int]:
    choices = list(range(1, max_distance + 1))
    upper = min(SMALL_COUNT_RANGE[1], max_distance - 1)
    lower = min(SMALL_COUNT_RANGE[0], upper)
    count = random.randint(lower, upper)
    for _ in range(200):
        distances = sorted(random.sample(choices, count))
        stretchable = [d for d in distances if d < max_distance and d + 1 not in distances]
        if stretchable and len(distances) < max_distance:
            return distances, random.choice(stretchable)
    raise RuntimeError("failed to choose ring distances")


def _generate_small_component(
    size: int,
    center: tuple[int, int],
    distance: int,
    area: int,
    occupied: set[tuple[int, int]],
    require_outer_cell: bool,
) -> set[tuple[int, int]] | None:
    ring_cells = _all_cells_on_distance(size, center, distance)
    random.shuffle(ring_cells)
    for anchor in ring_cells:
        if not _is_clear_from_occupied({anchor}, occupied, size):
            continue
        for _ in range(120):
            cells = {anchor}
            while len(cells) < area:
                frontier = set()
                for r, c in cells:
                    for dr, dc in NEIGHBORS4:
                        nr, nc = r + dr, c + dc
                        if (nr, nc) in cells:
                            continue
                        if not (0 <= nr < size and 0 <= nc < size):
                            continue
                        d = _chebyshev((nr, nc), center)
                        if d < distance or d > distance + 1:
                            continue
                        if not _is_clear_from_occupied({(nr, nc)}, occupied, size):
                            continue
                        frontier.add((nr, nc))
                if not frontier:
                    break
                if require_outer_cell and not any(
                    _chebyshev(cell, center) == distance + 1 for cell in cells
                ):
                    outer = [cell for cell in frontier if _chebyshev(cell, center) == distance + 1]
                    if outer:
                        frontier = set(outer)
                cells.add(random.choice(tuple(frontier)))
            if len(cells) != area:
                continue
            distances = {_chebyshev(cell, center) for cell in cells}
            if min(distances) != distance:
                continue
            if require_outer_cell and max(distances) == distance:
                continue
            return cells
    return None


def _place_noise_pixels(
    size: int,
    occupied: set[tuple[int, int]],
    colors: list[int],
) -> list[tuple[int, int, int]] | None:
    noises: list[tuple[int, int, int]] = []
    available = [(r, c) for r in range(size) for c in range(size)]
    random.shuffle(available)
    for color in colors:
        placed = False
        for r, c in available:
            if (r, c) in occupied:
                continue
            if not _is_clear_from_occupied({(r, c)}, occupied, size):
                continue
            noises.append((r, c, int(color)))
            occupied.add((r, c))
            placed = True
            break
        if not placed:
            return None
    return noises


def _render_input(
    size: int,
    large_cells: set[tuple[int, int]],
    small_components: list[tuple[int, set[tuple[int, int]]]],
    noises: list[tuple[int, int, int]],
) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    for r, c in large_cells:
        grid[r, c] = LARGE_COLOR
    for color, cells in small_components:
        for r, c in cells:
            grid[r, c] = int(color)
    for r, c, color in noises:
        grid[r, c] = int(color)
    return grid


def _distance_layers(comp: dict, center: tuple[int, int]) -> set[int]:
    return {_chebyshev((int(r), int(c)), center) for r, c in comp["coords"]}


def _semantic_output(
    grid: np.ndarray,
    center_override: tuple[int, int] | None = None,
    use_bbox_center: bool = False,
    use_farthest: bool = False,
    use_all_gray_pixels: bool = False,
) -> np.ndarray:
    bg, comps, large, small = _analyze(grid)
    if use_all_gray_pixels:
        gray_cells = np.argwhere(np.asarray(grid, dtype=int) == LARGE_COLOR)
        assert gray_cells.size > 0
        center = _centroid(gray_cells)
    elif center_override is not None:
        center = center_override
    elif use_bbox_center:
        center = _bbox_center(large["coords"])
    else:
        center = _centroid(large["coords"])

    mapping: dict[int, int] = {}
    for comp in small:
        distances = [_chebyshev((int(r), int(c)), center) for r, c in comp["coords"]]
        d = max(distances) if use_farthest else min(distances)
        mapping[d] = int(comp["color"])
    return _ring_output(grid.shape, center, mapping, bg)


def _components_are_separated(comps: list[dict], size: int) -> bool:
    owners: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(comps):
        for r, c in comp["coords"]:
            owners[(int(r), int(c))] = idx
    for idx, comp in enumerate(comps):
        for r, c in comp["coords"]:
            for dr, dc in NEIGHBORS8:
                nr, nc = int(r) + dr, int(c) + dc
                if not (0 <= nr < size and 0 <= nc < size):
                    continue
                if (nr, nc) in owners and owners[(nr, nc)] != idx:
                    return False
    return True


def _validate_generated_grid(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1]
    size = int(g.shape[0])
    assert size % 2 == 1 and size <= 30
    assert _background_color(g) == BG

    bg, comps, large, small = _analyze(g)
    assert bg == BG
    assert int(large["color"]) == LARGE_COLOR
    assert 10 <= int(large["area"]) <= 20
    assert SMALL_COUNT_RANGE[0] <= len(small) <= SMALL_COUNT_RANGE[1]
    assert all(SMALL_AREA_RANGE[0] <= int(comp["area"]) <= SMALL_AREA_RANGE[1] for comp in small)
    assert len({int(comp["color"]) for comp in small}) == len(small)
    assert LARGE_COLOR not in {int(comp["color"]) for comp in small}

    center = _centroid(large["coords"])
    assert center != (size // 2, size // 2), "anchor center must be off the geometric center"
    assert _bbox_center(large["coords"]) != center, "bbox-center shortcut must fail"

    distances = []
    has_stretched_component = False
    for comp in small:
        layers = _distance_layers(comp, center)
        distances.append(min(layers))
        has_stretched_component |= len(layers) > 1
    assert len(set(distances)) == len(distances)
    max_distance = max(max(center[0], size - 1 - center[0]), max(center[1], size - 1 - center[1]))
    assert len(set(distances)) < max_distance, "at least one distance layer must remain empty"
    assert has_stretched_component, "need a component spanning multiple layers"

    noise = [comp for comp in comps if int(comp["area"]) == 1]
    assert len(noise) >= 3
    gray_noise = [comp for comp in noise if int(comp["color"]) == LARGE_COLOR]
    assert len(gray_noise) >= 2, "need gray noise to invalidate color-only shortcuts"
    assert _components_are_separated(comps, size)

    correct = _semantic_output(g)
    assert np.array_equal(main(g), correct)
    assert not np.array_equal(correct, _semantic_output(g, center_override=(size // 2, size // 2)))
    assert not np.array_equal(correct, _semantic_output(g, use_bbox_center=True))
    assert not np.array_equal(correct, _semantic_output(g, use_farthest=True))
    assert not np.array_equal(correct, _semantic_output(g, use_all_gray_pixels=True))


def _attempt_generate() -> np.ndarray | None:
    size = random.choice(GRID_SIZES)
    center = _sample_center(size)
    large_cells = _generate_large_component(size, center)
    if large_cells is None:
        return None

    occupied = set(large_cells)
    max_distance = max(max(center[0], size - 1 - center[0]), max(center[1], size - 1 - center[1]))
    if max_distance < SMALL_COUNT_RANGE[0] + 1:
        return None

    distances, stretched_distance = _choose_distances(max_distance)
    colors = random.sample(SMALL_COLORS, len(distances))

    small_components: list[tuple[int, set[tuple[int, int]]]] = []
    for distance, color in sorted(zip(distances, colors), key=lambda x: -x[0]):
        area = random.randint(*SMALL_AREA_RANGE)
        require_outer = distance == stretched_distance or random.random() < 0.25
        cells = _generate_small_component(size, center, distance, area, occupied, require_outer)
        if cells is None:
            return None
        small_components.append((int(color), cells))
        occupied.update(cells)

    noise_colors = [LARGE_COLOR, LARGE_COLOR, small_components[0][0]]
    extra_noise_count = random.randint(NOISE_COUNT_RANGE[0], NOISE_COUNT_RANGE[1]) - len(noise_colors)
    extra_palette = [LARGE_COLOR] + [color for color, _cells in small_components] + list(SMALL_COLORS)
    noise_colors.extend(random.choice(extra_palette) for _ in range(max(0, extra_noise_count)))
    noises = _place_noise_pixels(size, occupied, noise_colors)
    if noises is None:
        return None

    grid = _render_input(size, large_cells, small_components, noises)
    try:
        _validate_generated_grid(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid centered-ring ARC input")
