import random
from collections import deque

import numpy as np


BG = 0
SOIL = 9

GEN_RETRIES = 400
GRID_H_RANGE = (15, 24)
GRID_W_RANGE = (15, 24)
SOIL_MARGIN = 4
SEED_COUNTS = (2, 3, 4)
SEED_COLORS = tuple(color for color in range(1, 10) if color != SOIL)
MIN_TERRITORY = 8
MIN_RECOLOR = 2

SIDES = ("top", "bottom", "left", "right")
CORNERS = ("TL", "TR", "BL", "BR")

TEMPLATE_LIBRARY = (
    ("##.", ".#.", "..."),
    ("#..", "###", "..."),
    ("#.#", ".#.", "#.."),
    ("###", "..#", "..."),
    (".#.", "###", ".#."),
    ("##.", "#..", ".#."),
    ("#..", "##.", "#.."),
    ("##.", ".##", "..."),
    ("#.#", "##.", "..."),
    ("..#", "###", "..."),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _largest_nonbg_component(grid: np.ndarray, bg: int):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    best_color = None
    best_coords = None
    best_area = -1

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if len(coords) > best_area:
                best_area = len(coords)
                best_color = color
                best_coords = coords

    assert best_coords is not None, "expected at least one foreground component"
    return best_color, best_coords


def _template_origin(coords: list[tuple[int, int]], h: int, w: int) -> tuple[int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]

    if max(rows) <= 2:
        r0 = 0
    elif min(rows) >= h - 3:
        r0 = h - 3
    else:
        raise AssertionError("template rows must lie in a corner 3x3 window")

    if max(cols) <= 2:
        c0 = 0
    elif min(cols) >= w - 3:
        c0 = w - 3
    else:
        raise AssertionError("template cols must lie in a corner 3x3 window")

    return r0, c0


def _parse_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    bg = _background_color(g)
    soil_color, soil_coords = _largest_nonbg_component(g, bg)
    soil_mask = np.zeros((h, w), dtype=bool)
    for r, c in soil_coords:
        soil_mask[r, c] = True

    seeds = []
    for r in range(h):
        for c in range(w):
            if soil_mask[r, c] or int(g[r, c]) == bg:
                continue
            if any(0 <= nr < h and 0 <= nc < w and soil_mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                seeds.append((r, c, int(g[r, c])))

    assert 2 <= len(seeds) <= 4, "expected 2 to 4 seeds"
    seed_colors = [color for _, _, color in seeds]
    assert len(set(seed_colors)) == len(seed_colors), "seed colors must be unique"
    seed_map = {color: (r, c) for r, c, color in seeds}

    templates: dict[int, np.ndarray] = {}
    for color, seed_pos in seed_map.items():
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        template_cells = [rc for rc in coords if rc != seed_pos]
        assert template_cells, "every seed color must have a matching template"

        r0, c0 = _template_origin(template_cells, h, w)
        mask = np.zeros((3, 3), dtype=bool)
        for r, c in template_cells:
            rr = r - r0
            cc = c - c0
            assert 0 <= rr < 3 and 0 <= cc < 3, "template pixels must fit inside a 3x3 window"
            mask[rr, cc] = True
        templates[color] = mask

    return bg, soil_color, soil_mask, seeds, templates


def _assign_territories(soil_mask: np.ndarray, seeds: list[tuple[int, int, int]]):
    owners: dict[tuple[int, int], tuple[int, int, int]] = {}
    counts = {color: 0 for _, _, color in seeds}
    for r, c in np.argwhere(soil_mask):
        rr = int(r)
        cc = int(c)
        owner = min(
            seeds,
            key=lambda item: (max(abs(rr - item[0]), abs(cc - item[1])), item[0], item[1]),
        )
        owners[(rr, cc)] = owner
        counts[owner[2]] += 1
    return owners, counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partition the soil by nearest seed (Chebyshev distance), then tile each
    territory with the matching corner template anchored at the seed position.
    """
    g = np.asarray(grid, dtype=int)
    _, soil_color, soil_mask, seeds, templates = _parse_scene(g)

    owners, _ = _assign_territories(soil_mask, seeds)
    out = g.copy()

    for (r, c), (sr, sc, color) in owners.items():
        mask = templates[color]
        if mask[(r - sr) % 3, (c - sc) % 3]:
            out[r, c] = color
        else:
            out[r, c] = soil_color

    return out


def _strings_to_mask(rows: tuple[str, str, str]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


def _random_template(rng: random.Random) -> np.ndarray:
    mask = _strings_to_mask(rng.choice(TEMPLATE_LIBRARY))
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    mask = np.rot90(mask, rng.randrange(4))
    return mask.copy()


def _generate_intervals(height: int, width: int, rng: random.Random) -> list[tuple[int, int]]:
    min_row_width = max(4, width // 2)
    left = rng.randint(0, max(0, width - min_row_width))
    right = rng.randint(left + min_row_width - 1, width - 1)
    intervals = []

    for _ in range(height):
        intervals.append((left, right))
        for _ in range(20):
            new_left = left + rng.choice((-2, -1, 0, 1, 2))
            new_right = right + rng.choice((-2, -1, 0, 1, 2))
            new_left = max(0, min(new_left, width - 4))
            new_right = min(width - 1, max(new_right, new_left + 3))
            if new_right - new_left + 1 < min_row_width:
                continue
            if new_left <= right and new_right >= left:
                left, right = new_left, new_right
                break

    return intervals


def _shape_is_irregular(intervals: list[tuple[int, int]]) -> bool:
    lefts = {left for left, _ in intervals}
    rights = {right for _, right in intervals}
    return len(lefts) > 1 or len(rights) > 1


def _fill_soil(grid: np.ndarray, r0: int, c0: int, intervals: list[tuple[int, int]]):
    for dr, (left, right) in enumerate(intervals):
        grid[r0 + dr, c0 + left : c0 + right + 1] = SOIL


def _has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()

    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    return bool(np.any(~mask & ~outside))


def _central_band(values: list[int]) -> list[int]:
    if len(values) <= 2:
        return values
    trim = max(1, len(values) // 4)
    band = values[trim : len(values) - trim]
    return band or values


def _pick_seed_position(side: str, r0: int, c0: int, intervals: list[tuple[int, int]], rng: random.Random):
    height = len(intervals)
    if side == "top":
        left, right = intervals[0]
        cols = _central_band(list(range(left, right + 1)))
        return r0 - 1, c0 + rng.choice(cols)
    if side == "bottom":
        left, right = intervals[-1]
        cols = _central_band(list(range(left, right + 1)))
        return r0 + height, c0 + rng.choice(cols)
    rows = _central_band(list(range(height)))
    row_idx = rng.choice(rows)
    left, right = intervals[row_idx]
    if side == "left":
        return r0 + row_idx, c0 + left - 1
    if side == "right":
        return r0 + row_idx, c0 + right + 1
    raise ValueError(f"unknown side: {side}")


def _corner_origin(name: str, h: int, w: int) -> tuple[int, int]:
    if name == "TL":
        return 0, 0
    if name == "TR":
        return 0, w - 3
    if name == "BL":
        return h - 3, 0
    if name == "BR":
        return h - 3, w - 3
    raise ValueError(f"unknown corner: {name}")


def _place_template(grid: np.ndarray, corner: str, color: int, mask: np.ndarray):
    r0, c0 = _corner_origin(corner, *grid.shape)
    rows, cols = np.where(mask)
    for rr, cc in zip(rows.tolist(), cols.tolist()):
        grid[r0 + rr, c0 + cc] = color


def _gen_attempt(rng: random.Random):
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    seed_count = rng.choice(SEED_COUNTS)

    soil_h = rng.randint(6, h - 2 * SOIL_MARGIN)
    soil_w = rng.randint(6, w - 2 * SOIL_MARGIN)
    r0 = rng.randint(SOIL_MARGIN, h - soil_h - SOIL_MARGIN)
    c0 = rng.randint(SOIL_MARGIN, w - soil_w - SOIL_MARGIN)

    intervals = _generate_intervals(soil_h, soil_w, rng)
    if not _shape_is_irregular(intervals):
        return None

    grid = np.full((h, w), BG, dtype=int)
    _fill_soil(grid, r0, c0, intervals)
    soil_mask = grid == SOIL
    if int(np.count_nonzero(soil_mask)) < seed_count * MIN_TERRITORY:
        return None
    if _has_hole(soil_mask):
        return None

    chosen_sides = rng.sample(SIDES, seed_count)
    seed_positions = [_pick_seed_position(side, r0, c0, intervals, rng) for side in chosen_sides]
    if len(set(seed_positions)) != seed_count:
        return None
    if any(max(abs(r1 - r2), abs(c1 - c2)) < 3 for i, (r1, c1) in enumerate(seed_positions) for r2, c2 in seed_positions[i + 1 :]):
        return None

    colors = rng.sample(SEED_COLORS, seed_count)
    for (r, c), color in zip(seed_positions, colors):
        grid[r, c] = color

    chosen_corners = rng.sample(CORNERS, seed_count)
    masks = {color: _random_template(rng) for color in colors}
    for color, corner in zip(colors, chosen_corners):
        _place_template(grid, corner, color, masks[color])

    try:
        _, _, parsed_soil_mask, parsed_seeds, _ = _parse_scene(grid)
    except AssertionError:
        return None

    if len(parsed_seeds) != seed_count:
        return None

    _, territory_sizes = _assign_territories(parsed_soil_mask, parsed_seeds)
    if any(size < MIN_TERRITORY for size in territory_sizes.values()):
        return None

    out = main(grid)
    if int(np.count_nonzero(out != grid)) < seed_count * MIN_RECOLOR:
        return None

    for color in colors:
        recolored = np.count_nonzero((out == color) & parsed_soil_mask)
        if recolored < MIN_RECOLOR:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
