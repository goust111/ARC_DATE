import random

import numpy as np


BG = 0
DIRS = ("up", "right", "down", "left")
ENTRY_HEIGHT = 2
ENTRY_WIDTH = 4
ENTRY_GAP = 1
LEGEND_ROWS = 4
LEGEND_HEIGHT = LEGEND_ROWS * ENTRY_HEIGHT + (LEGEND_ROWS - 1) * ENTRY_GAP
LEGEND_WIDTH = ENTRY_WIDTH
OBJECT_COUNT_RANGE = (2, 4)
OBJECT_HEIGHT_RANGE = (3, 7)
OBJECT_WIDTH_RANGE = (3, 7)
GRID_HEIGHT_RANGE = (13, 22)
GRID_WIDTH_RANGE = (15, 27)
OBJECT_LEFT_MARGIN = LEGEND_WIDTH + 2
GEN_RETRIES = 300
PLACEMENT_RETRIES = 200

INDICATOR_PATTERNS = {
    "up": np.array([[1, 1], [1, 0]], dtype=int),
    "right": np.array([[1, 1], [0, 1]], dtype=int),
    "down": np.array([[0, 1], [1, 1]], dtype=int),
    "left": np.array([[1, 0], [1, 1]], dtype=int),
}


def _mode_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel())
    return int(np.argmax(counts))


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in np.asarray(mask, dtype=int))


PATTERN_TO_DIR = {_mask_key(mask): direction for direction, mask in INDICATOR_PATTERNS.items()}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _parse_legend_entry(grid: np.ndarray, coords: np.ndarray, bg: int) -> tuple[str, int]:
    r0, c0, r1, c1 = _bbox(coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert c0 == 0, "legend entries must touch the left edge"
    assert sub.shape == (ENTRY_HEIGHT, ENTRY_WIDTH), "legend entry size mismatch"

    colors = [int(color) for color in np.unique(sub) if int(color) != bg]
    assert len(colors) == 2, "legend entry must contain indicator and swatch colors"
    counts = {color: int(np.sum(sub == color)) for color in colors}
    indicator_color = next((color for color, count in counts.items() if count == 3), None)
    swatch_color = next((color for color, count in counts.items() if count == 4), None)
    assert indicator_color is not None and swatch_color is not None, "legend entry color counts must be 3 and 4"

    indicator_mask = (sub[:, :2] == indicator_color).astype(int)
    swatch_mask = sub[:, 2:] == swatch_color
    assert np.all(sub[:, :2][indicator_mask == 0] == bg), "indicator must live in the left 2x2 block"
    assert np.all(swatch_mask), "swatch must be a solid 2x2 block"

    direction = PATTERN_TO_DIR.get(_mask_key(indicator_mask))
    assert direction is not None, "unrecognized legend indicator"
    return direction, swatch_color


def _legend_mapping(grid: np.ndarray, bg: int) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for coords in _components(grid != bg):
        if int(coords[:, 1].min()) != 0:
            continue
        direction, color = _parse_legend_entry(grid, coords, bg)
        assert direction not in mapping, "each direction must appear once in the legend"
        mapping[direction] = color
    assert set(mapping) == set(DIRS), "legend must provide all four directions"
    return mapping


def _object_direction_and_hole(grid: np.ndarray, coords: np.ndarray, bg: int) -> tuple[str, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    sub_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    sub_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    holes = np.argwhere(~sub_mask)
    assert len(holes) == 1, "each object must have exactly one single-cell notch"
    hr, hc = map(int, holes[0])

    on_top = hr == 0
    on_bottom = hr == sub_mask.shape[0] - 1
    on_left = hc == 0
    on_right = hc == sub_mask.shape[1] - 1
    assert sum((on_top, on_bottom, on_left, on_right)) == 1, "notch must lie on exactly one bbox edge"

    if on_top:
        direction = "up"
    elif on_right:
        direction = "right"
    elif on_bottom:
        direction = "down"
    else:
        direction = "left"

    return direction, (r0 + hr, c0 + hc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each object's single-cell notch using the color chosen by the left-edge direction legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG
    mapping = _legend_mapping(g, bg)

    out = g.copy()
    for coords in _components(g != bg):
        if int(coords[:, 1].min()) == 0:
            continue
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "foreground objects must be monochrome"
        direction, hole = _object_direction_and_hole(g, coords, bg)
        out[hole] = mapping[direction]
    return out


def _legend_entry(indicator_color: int, fill_color: int, direction: str) -> np.ndarray:
    patch = np.zeros((ENTRY_HEIGHT, ENTRY_WIDTH), dtype=int)
    patch[:, :2] = INDICATOR_PATTERNS[direction] * indicator_color
    patch[:, 2:] = fill_color
    return patch


def _object_patch(color: int, height: int, width: int, direction: str, notch_index: int) -> tuple[np.ndarray, tuple[int, int]]:
    patch = np.full((height, width), color, dtype=int)
    if direction == "up":
        hole = (0, notch_index)
    elif direction == "right":
        hole = (notch_index, width - 1)
    elif direction == "down":
        hole = (height - 1, notch_index)
    else:
        hole = (notch_index, 0)
    patch[hole] = BG
    return patch, hole


def _expected_output(grid: np.ndarray, objects: list[dict[str, object]], mapping: dict[str, int]) -> np.ndarray:
    out = grid.copy()
    for obj in objects:
        r, c = obj["hole"]
        out[int(r), int(c)] = int(mapping[obj["direction"]])
    return out


def _reserve(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    h, w = blocked.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    if height < LEGEND_HEIGHT or width <= OBJECT_LEFT_MARGIN + 2:
        return None

    num_objects = random.randint(*OBJECT_COUNT_RANGE)
    palette = random.sample(list(range(1, 10)), 1 + len(DIRS) + num_objects)
    indicator_color = palette[0]
    fill_colors = palette[1 : 1 + len(DIRS)]
    object_colors = palette[1 + len(DIRS) :]

    shuffled_fills = fill_colors[:]
    random.shuffle(shuffled_fills)
    direction_to_fill = dict(zip(DIRS, shuffled_fills))

    grid = np.zeros((height, width), dtype=int)
    legend_top = random.randint(0, height - LEGEND_HEIGHT)
    legend_order = random.sample(list(DIRS), len(DIRS))
    for idx, direction in enumerate(legend_order):
        r0 = legend_top + idx * (ENTRY_HEIGHT + ENTRY_GAP)
        grid[r0 : r0 + ENTRY_HEIGHT, 0:ENTRY_WIDTH] = _legend_entry(
            indicator_color, direction_to_fill[direction], direction
        )

    blocked = np.zeros_like(grid, dtype=bool)
    _reserve(blocked, legend_top, 0, legend_top + LEGEND_HEIGHT - 1, LEGEND_WIDTH - 1)
    blocked[:, :OBJECT_LEFT_MARGIN] = True

    directions = random.sample(list(DIRS), num_objects)
    random.shuffle(directions)
    objects: list[dict[str, object]] = []

    for color, direction in zip(object_colors, directions):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            obj_h = random.randint(*OBJECT_HEIGHT_RANGE)
            obj_w = random.randint(*OBJECT_WIDTH_RANGE)
            if height < obj_h or width - OBJECT_LEFT_MARGIN < obj_w:
                continue

            if direction in ("up", "down"):
                notch_index = random.randint(1, obj_w - 2)
            else:
                notch_index = random.randint(1, obj_h - 2)
            patch, local_hole = _object_patch(color, obj_h, obj_w, direction, notch_index)

            r0 = random.randint(0, height - obj_h)
            c0 = random.randint(OBJECT_LEFT_MARGIN, width - obj_w)
            if np.any(blocked[r0 : r0 + obj_h, c0 : c0 + obj_w]):
                continue

            grid[r0 : r0 + obj_h, c0 : c0 + obj_w] = patch
            _reserve(blocked, r0, c0, r0 + obj_h - 1, c0 + obj_w - 1)
            objects.append(
                {
                    "direction": direction,
                    "hole": (r0 + local_hole[0], c0 + local_hole[1]),
                }
            )
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(grid, objects, direction_to_fill)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
