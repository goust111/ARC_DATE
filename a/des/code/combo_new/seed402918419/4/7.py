from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
GRAY = 5

GEN_COLOR_POOL = (2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ("up", "down", "left", "right")
GEN_RETRIES = 300
PATTERN_RETRIES = 200


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in np.argwhere(mask)))


_ARROW_BASE_UP = np.array(
    [
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ],
    dtype=bool,
)

ARROW_MASKS = {
    "up": _ARROW_BASE_UP,
    "down": np.rot90(_ARROW_BASE_UP, 2),
    "left": np.rot90(_ARROW_BASE_UP, 1),
    "right": np.rot90(_ARROW_BASE_UP, 3),
}
ARROW_SIGNATURES = {
    _shape_signature(mask): direction for direction, mask in ARROW_MASKS.items()
}

_RNG = random.Random()


def _apply_transform(arr: np.ndarray, direction: str) -> np.ndarray:
    if direction == "up":
        return arr.copy()
    if direction == "down":
        return np.flipud(arr)
    if direction == "left":
        return np.fliplr(arr)
    if direction == "right":
        return np.rot90(arr, 2)
    raise ValueError(f"unknown direction: {direction}")


def _frame_bbox(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == color)
    assert len(coords) > 0, f"missing frame color {color}"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a non-empty interior"

    expected: set[tuple[int, int]] = set()
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))

    actual = {(int(r), int(c)) for r, c in coords}
    assert actual == expected, f"color {color} must form a single rectangular border"
    return r0, c0, r1, c1


def _inside_bbox(coord: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = coord
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _components_of_color(
    grid: np.ndarray, color: int, connectivity: int = 8
) -> list[list[tuple[int, int]]]:
    mask = grid == color
    h, w = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 8:
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in neighbors:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _normalized_signature(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _marker_order(
    grid: np.ndarray, source_bbox: tuple[int, int, int, int]
) -> list[int]:
    h, w = grid.shape
    r0, c0, r1, c1 = source_bbox
    candidates: list[list[int]] = []
    for marker_c in (c0 - 1, c1 + 1):
        if not (0 <= marker_c < w):
            continue
        colors = [
            int(grid[r, marker_c])
            for r in range(r0 + 1, r1)
            if int(grid[r, marker_c]) not in {BG, BLUE, GRAY}
        ]
        if colors:
            assert len(colors) == len(set(colors)), "marker colors must be unique"
            candidates.append(colors)
    assert len(candidates) == 1, "expected exactly one external marker column"
    return candidates[0]


def _arrow_directions(
    grid: np.ndarray,
    colors: list[int],
    source_bbox: tuple[int, int, int, int],
    target_bbox: tuple[int, int, int, int],
) -> dict[int, str]:
    result: dict[int, str] = {}
    for color in colors:
        directions: list[str] = []
        for comp in _components_of_color(grid, color, connectivity=8):
            if any(_inside_bbox(pt, source_bbox) or _inside_bbox(pt, target_bbox) for pt in comp):
                continue
            signature = _normalized_signature(comp)
            direction = ARROW_SIGNATURES.get(signature)
            if direction is not None:
                directions.append(direction)
        assert len(directions) == 1, f"expected one arrow for color {color}"
        result[int(color)] = directions[0]
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform the source pattern by arrow direction and stack color-selected layers into the target canvas.

    Input structure:
    - The background is black.
    - One blue rectangular frame encloses the source pattern, whose interior uses several non-black, non-blue colors.
    - One gray rectangular frame encloses a same-sized black target canvas.
    - A single external marker column beside the blue frame lists the arrow colors from top to bottom.
    - Each arrow is a small monochrome V-headed component whose opening direction is one of up, down, left, or right.

    Rule:
    - Read the source pattern from inside the blue frame and the marker order from the adjacent color column.
    - For each arrow color, locate the matching arrow and map its pointing direction to a transform: up keeps the source unchanged, down flips vertically, left flips horizontally, and right rotates 180 degrees.
    - Apply every transform to the whole source pattern, but keep only the pixels whose color matches that arrow's color.
    - Paint those color-selected layers into the gray target interior from lowest priority to highest priority, where the marker column's top cell is highest priority and later paints overwrite earlier ones.
    - Keep the source frame, arrows, marker column, and every non-target cell unchanged.

    Key insight:
    - The arrows do not copy themselves; each one chooses a geometric transform for the entire source image, while the nearby marker column separately encodes the layer priority that resolves overlaps between transformed same-color slices.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    source_bbox = _frame_bbox(g, BLUE)
    target_bbox = _frame_bbox(g, GRAY)
    sr0, sc0, sr1, sc1 = source_bbox
    tr0, tc0, tr1, tc1 = target_bbox

    source = g[sr0 + 1 : sr1, sc0 + 1 : sc1]
    assert source.size > 0, "source interior must be non-empty"
    assert source.shape == (tr1 - tr0 - 1, tc1 - tc0 - 1), "target interior size must match source interior size"

    marker_order = _marker_order(g, source_bbox)
    arrow_dirs = _arrow_directions(g, marker_order, source_bbox, target_bbox)

    target = np.full(source.shape, BG, dtype=int)
    for color in reversed(marker_order):
        transformed = _apply_transform(source, arrow_dirs[color])
        mask = transformed == color
        target[mask] = int(color)

    out = g.copy()
    out[tr0 + 1 : tr1, tc0 + 1 : tc1] = target
    return out


def _paint_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _paint_arrow(grid: np.ndarray, r0: int, c0: int, direction: str, color: int) -> None:
    mask = ARROW_MASKS[direction]
    h, w = mask.shape
    grid[r0 : r0 + h, c0 : c0 + w][mask] = color


def _slot_anchor(
    source_r0: int,
    source_c0: int,
    outer_h: int,
    outer_w: int,
    slot_name: str,
    direction: str,
) -> tuple[int, int]:
    mask = ARROW_MASKS[direction]
    ah, aw = mask.shape
    center_r = source_r0 + outer_h // 2
    center_c = source_c0 + outer_w // 2

    if slot_name == "north":
        return source_r0 - ah - 1, center_c - aw // 2
    if slot_name == "south":
        return source_r0 + outer_h + 1, center_c - aw // 2
    if slot_name == "west":
        return center_r - ah // 2, source_c0 - aw - 2
    if slot_name == "east":
        return center_r - ah // 2, source_c0 + outer_w + 2
    raise ValueError(f"unknown slot: {slot_name}")


def _contention_exists(
    source: np.ndarray, color_to_direction: dict[int, str]
) -> bool:
    contenders = np.zeros(source.shape, dtype=int)
    for color, direction in color_to_direction.items():
        transformed = _apply_transform(source, direction)
        contenders += (transformed == color).astype(int)
    return bool(np.any(contenders >= 2))


def _pattern_ok(source: np.ndarray, color_to_direction: dict[int, str]) -> bool:
    for color, direction in color_to_direction.items():
        if int(np.count_nonzero(source == color)) < 3:
            return False
        if direction != "up":
            original = source == color
            moved = _apply_transform(original.astype(int), direction) > 0
            if np.array_equal(original, moved):
                return False
    if not _contention_exists(source, color_to_direction):
        return False
    return len(np.unique(source)) == len(color_to_direction)


def _make_source_pattern(
    rng: random.Random,
    inner_h: int,
    inner_w: int,
    colors: list[int],
    color_to_direction: dict[int, str],
) -> np.ndarray | None:
    cell_count = inner_h * inner_w
    for _ in range(PATTERN_RETRIES):
        cells = [rng.choice(colors) for _ in range(cell_count)]
        source = np.array(cells, dtype=int).reshape(inner_h, inner_w)
        if _pattern_ok(source, color_to_direction):
            return source
    return None


def _target_positions(
    grid: np.ndarray, outer_h: int, outer_w: int
) -> list[tuple[int, int]]:
    h, w = grid.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - outer_h - 1):
        for c0 in range(1, w - outer_w - 1):
            region = grid[r0 - 1 : r0 + outer_h + 1, c0 - 1 : c0 + outer_w + 1]
            if np.any(region != BG):
                continue
            positions.append((r0, c0))
    return positions


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    inner_h = rng.randint(4, 7)
    inner_w = rng.randint(4, 7)
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    grid_h = rng.randint(max(26, outer_h + 12), 30)
    grid_w = rng.randint(max(26, outer_w + 14), 30)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    colors = rng.sample(GEN_COLOR_POOL, 4)
    color_to_direction = {
        int(color): direction for color, direction in zip(colors, rng.sample(list(DIRECTIONS), 4))
    }
    priority_order = rng.sample(colors, len(colors))

    source = _make_source_pattern(rng, inner_h, inner_w, colors, color_to_direction)
    if source is None:
        return None

    source_r0_min = 7
    source_c0_min = 8
    source_r0_max = grid_h - outer_h - 7
    source_c0_max = grid_w - outer_w - 8
    if source_r0_min > source_r0_max or source_c0_min > source_c0_max:
        return None
    source_r0 = rng.randint(source_r0_min, source_r0_max)
    source_c0 = rng.randint(source_c0_min, source_c0_max)
    source_r1 = source_r0 + outer_h - 1
    source_c1 = source_c0 + outer_w - 1

    _paint_frame(grid, source_r0, source_c0, outer_h, outer_w, BLUE)
    grid[source_r0 + 1 : source_r1, source_c0 + 1 : source_c1] = source

    marker_side = rng.choice(("left", "right"))
    marker_c = source_c0 - 1 if marker_side == "left" else source_c1 + 1
    marker_start = source_r0 + 1 + rng.randint(0, inner_h - len(priority_order))
    for idx, color in enumerate(priority_order):
        grid[marker_start + idx, marker_c] = int(color)

    arrow_slots = ("north", "south", "west", "east")
    pair_order = list(color_to_direction.items())
    rng.shuffle(pair_order)
    for slot_name, (color, direction) in zip(arrow_slots, pair_order):
        r0, c0 = _slot_anchor(source_r0, source_c0, outer_h, outer_w, slot_name, direction)
        _paint_arrow(grid, r0, c0, direction, int(color))

    target_positions = _target_positions(grid, outer_h, outer_w)
    if not target_positions:
        return None
    target_r0, target_c0 = rng.choice(target_positions)
    _paint_frame(grid, target_r0, target_c0, outer_h, outer_w, GRAY)

    out = main(grid)
    target = out[target_r0 + 1 : target_r0 + outer_h - 1, target_c0 + 1 : target_c0 + outer_w - 1]
    if np.count_nonzero(target) < inner_h:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
