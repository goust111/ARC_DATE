from __future__ import annotations

import numpy as np


BG = 8
COLOR_CYCLE = (2, 1, 3, 4)  # red -> blue -> green -> yellow -> red
COLOR_INDEX = {color: i for i, color in enumerate(COLOR_CYCLE)}

# Generator settings.
ODD_SIZES = (11, 13, 15, 17, 19)
EXTRA_OBJECTS_RANGE = (0, 3)
MAX_SHAPE_SIDE = 4
MAX_SHAPE_AREA = 6
PLACEMENT_TRIES = 80
GENERATION_RETRIES = 400


def _find_components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            rows = arr[:, 0]
            cols = arr[:, 1]
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())),
                }
            )

    return components


def _quadrant_offset(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> int:
    r0, c0, r1, c1 = bbox
    h, w = shape
    row_sum = r0 + r1
    col_sum = c0 + c1
    center_row_sum = h - 1
    center_col_sum = w - 1

    assert row_sum != center_row_sum and col_sum != center_col_sum, "sprite center must lie in one strict quadrant"

    top = row_sum < center_row_sum
    left = col_sum < center_col_sum
    if top and left:
        return 0
    if top and not left:
        return 1
    if not top and not left:
        return 2
    return 3


def _advance_color(color: int, steps: int) -> int:
    assert color in COLOR_INDEX, "unexpected sprite color"
    return COLOR_CYCLE[(COLOR_INDEX[color] + steps) % len(COLOR_CYCLE)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each sprite by a quadrant-dependent shift along the cycle
    red -> blue -> green -> yellow -> red.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    components = _find_components(g)
    assert components, "expected at least one foreground sprite"

    out = g.copy()
    for component in components:
        coords = component["coords"]
        color = int(component["color"])
        offset = _quadrant_offset(component["bbox"], g.shape)
        out[coords[:, 0], coords[:, 1]] = _advance_color(color, offset)
    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_connected_mask(rng: np.random.Generator, max_h: int, max_w: int) -> np.ndarray:
    raw_h = int(rng.integers(1, min(MAX_SHAPE_SIDE, max_h) + 1))
    raw_w = int(rng.integers(1, min(MAX_SHAPE_SIDE, max_w) + 1))
    max_area = min(MAX_SHAPE_AREA, raw_h * raw_w)
    min_area = 1 if max_area == 1 else 2
    area = int(rng.integers(min_area, max_area + 1))

    start = (int(rng.integers(raw_h)), int(rng.integers(raw_w)))
    cells = {start}
    while len(cells) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < raw_h and 0 <= nc < raw_w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        pick = list(frontier)[int(rng.integers(len(frontier)))]
        cells.add(pick)

    mask = np.zeros((raw_h, raw_w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _trim_mask(mask)


def _quadrant_region(shape: tuple[int, int], quadrant: int) -> tuple[int, int, int, int]:
    h, w = shape
    center_r = h // 2
    center_c = w // 2
    if quadrant == 0:
        return 0, center_r, 0, center_c
    if quadrant == 1:
        return 0, center_r, center_c + 1, w
    if quadrant == 2:
        return center_r + 1, h, center_c + 1, w
    return center_r + 1, h, 0, center_c


def _can_place(grid: np.ndarray, coords: np.ndarray) -> bool:
    h, w = grid.shape
    cells = {(int(r), int(c)) for r, c in coords}
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != BG:
            return False
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells and int(grid[nr, nc]) != BG:
                    return False
    return True


def _place_object(grid: np.ndarray, rng: np.random.Generator, quadrant: int, color: int) -> bool:
    r_lo, r_hi, c_lo, c_hi = _quadrant_region(grid.shape, quadrant)
    region_h = r_hi - r_lo
    region_w = c_hi - c_lo
    for _ in range(PLACEMENT_TRIES):
        mask = _random_connected_mask(rng, region_h, region_w)
        patch_h, patch_w = mask.shape
        top = int(rng.integers(r_lo, r_hi - patch_h + 1))
        left = int(rng.integers(c_lo, c_hi - patch_w + 1))
        local = np.argwhere(mask)
        coords = local + np.array([top, left], dtype=int)
        if not _can_place(grid, coords):
            continue
        grid[coords[:, 0], coords[:, 1]] = color
        return True
    return False


def _has_all_quadrants(grid: np.ndarray) -> bool:
    quadrants = {_quadrant_offset(component["bbox"], grid.shape) for component in _find_components(grid)}
    return quadrants == {0, 1, 2, 3}


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.choice(ODD_SIZES))
    w = int(rng.choice(ODD_SIZES))
    grid = np.full((h, w), BG, dtype=int)

    specs = [(quadrant, int(color)) for quadrant, color in enumerate(rng.permutation(COLOR_CYCLE))]
    extra_count = int(rng.integers(EXTRA_OBJECTS_RANGE[0], EXTRA_OBJECTS_RANGE[1] + 1))
    for _ in range(extra_count):
        specs.append((int(rng.integers(0, 4)), int(rng.choice(COLOR_CYCLE))))
    rng.shuffle(specs)

    for quadrant, color in specs:
        if not _place_object(grid, rng, quadrant, color):
            return None

    components = _find_components(grid)
    if len(components) != len(specs):
        return None
    if set(int(component["color"]) for component in components) != set(COLOR_CYCLE):
        return None
    if not _has_all_quadrants(grid):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_RETRIES} attempts")
