import random
from collections import Counter

import numpy as np


DIRECTIONS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_HEIGHT_RANGE = (18, 24)
GRID_WIDTH_RANGE = (18, 24)
OBJECT_COUNT_RANGE = (2, 4)
SHAPE_SIZES = ((3, 3), (3, 4), (4, 3), (4, 4))
MAX_EXTRA_HOLES = 2
MAX_GENERATION_ATTEMPTS = 500


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _legend_mapping(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    assert h >= 3, "expected room for a two-row bottom legend"

    active = np.flatnonzero((grid[h - 2] != bg) | (grid[h - 1] != bg))
    assert active.size > 0, "expected a bottom legend"

    c0 = int(active.min())
    c1 = int(active.max()) + 1

    assert not np.any(grid[h - 2 :, :c0] != bg)
    assert not np.any(grid[h - 2 :, c1:] != bg)

    upper = grid[h - 2, c0:c1]
    lower = grid[h - 1, c0:c1]
    assert np.all(upper != bg) and np.all(lower != bg), "legend entries must be complete vertical pairs"

    mapping: dict[int, int] = {}
    for src, dst in zip(upper.tolist(), lower.tolist(), strict=False):
        src = int(src)
        dst = int(dst)
        if src in mapping:
            assert mapping[src] == dst, "legend must not give conflicting mappings"
        mapping[src] = dst

    mask = np.zeros((h, w), dtype=bool)
    mask[h - 2 : h, c0:c1] = True
    return mapping, mask


def _find_components(grid: np.ndarray, bg: int, blocked: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or blocked[r0, c0] or int(grid[r0, c0]) == bg:
                continue

            color = int(grid[r0, c0])
            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells: list[tuple[int, int]] = []

            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and not blocked[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            components.append(
                {
                    "color": color,
                    "coords": np.array(cells, dtype=int),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    return components


def _gap_info(grid: np.ndarray, component: dict[str, object], bg: int) -> tuple[str, tuple[int, int]]:
    color = int(component["color"])
    r0, c0, r1, c1 = component["bbox"]
    gaps: list[tuple[str, tuple[int, int]]] = []

    for c in range(c0 + 1, c1):
        if int(grid[r0, c]) == bg and int(grid[r0, c - 1]) == color and int(grid[r0, c + 1]) == color:
            gaps.append(("top", (r0, c)))
        if int(grid[r1, c]) == bg and int(grid[r1, c - 1]) == color and int(grid[r1, c + 1]) == color:
            gaps.append(("bottom", (r1, c)))

    for r in range(r0 + 1, r1):
        if int(grid[r, c0]) == bg and int(grid[r - 1, c0]) == color and int(grid[r + 1, c0]) == color:
            gaps.append(("left", (r, c0)))
        if int(grid[r, c1]) == bg and int(grid[r - 1, c1]) == color and int(grid[r + 1, c1]) == color:
            gaps.append(("right", (r, c1)))

    assert len(gaps) == 1, "each object must have exactly one bbox-side gap"
    return gaps[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each foreground object's area into a ray whose direction comes from its bbox-side gap.

    The bottom two rows form a legend: the upper legend row lists object colors and the
    lower row lists the corresponding ray colors. Each foreground object is monochromatic,
    4-connected, and has exactly one gap on one side of its bounding box. The gap direction
    determines the ray direction, and the object area determines the ray length.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    mapping, legend_mask = _legend_mapping(g, bg)
    components = _find_components(g, bg, legend_mask)

    out = g.copy()
    for component in components:
        color = int(component["color"])
        assert color in mapping, "every object color must appear in the legend"

        direction, start = _gap_info(g, component, bg)
        dr, dc = DIRECTIONS[direction]
        ray_color = int(mapping[color])
        length = int(component["coords"].shape[0])

        for step in range(length):
            rr = int(start[0] + dr * step)
            cc = int(start[1] + dc * step)
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and int(out[rr, cc]) == bg:
                out[rr, cc] = ray_color

    return out


def _make_shape(direction: str) -> dict[str, object]:
    h, w = random.choice(SHAPE_SIZES)
    mask = np.ones((h, w), dtype=bool)

    if direction == "top":
        gap = (0, random.randint(1, w - 2))
    elif direction == "bottom":
        gap = (h - 1, random.randint(1, w - 2))
    elif direction == "left":
        gap = (random.randint(1, h - 2), 0)
    else:
        gap = (random.randint(1, h - 2), w - 1)

    mask[gap] = False

    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    max_holes = min(MAX_EXTRA_HOLES, len(interior))
    hole_count = random.randint(0, max_holes)
    for cell in random.sample(interior, hole_count):
        mask[cell] = False

    return {
        "direction": direction,
        "mask": mask,
        "gap": gap,
        "area": int(mask.sum()),
        "height": h,
        "width": w,
    }


def _object_cells(mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    return np.argwhere(mask) + np.array([r0, c0], dtype=int)


def _ray_cells(spec: dict[str, object], r0: int, c0: int) -> list[tuple[int, int]]:
    gap_r, gap_c = spec["gap"]
    start_r = int(r0 + gap_r)
    start_c = int(c0 + gap_c)
    dr, dc = DIRECTIONS[str(spec["direction"])]
    return [
        (start_r + dr * step, start_c + dc * step)
        for step in range(int(spec["area"]))
    ]


def _position_ranges(spec: dict[str, object], play_h: int, width: int) -> tuple[range, range]:
    h = int(spec["height"])
    w = int(spec["width"])
    area = int(spec["area"])
    direction = str(spec["direction"])

    if direction == "top":
        row_range = range(area - 1, play_h - h + 1)
        col_range = range(0, width - w + 1)
    elif direction == "bottom":
        row_range = range(0, play_h - h - area + 2)
        col_range = range(0, width - w + 1)
    elif direction == "left":
        row_range = range(0, play_h - h + 1)
        col_range = range(area - 1, width - w + 1)
    else:
        row_range = range(0, play_h - h + 1)
        col_range = range(0, width - w - area + 2)

    return row_range, col_range


def _coords_fit(coords: np.ndarray, mask: np.ndarray) -> bool:
    for r, c in coords:
        r = int(r)
        c = int(c)
        if mask[r, c]:
            return False
    return True


def _touches_existing_object(coords: np.ndarray, object_mask: np.ndarray) -> bool:
    h, w = object_mask.shape
    for r, c in coords:
        r = int(r)
        c = int(c)
        if object_mask[r, c]:
            return True
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and object_mask[nr, nc]:
                return True
    return False


def _paint_cells(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    play_h = height - 2

    object_count = random.randint(*OBJECT_COUNT_RANGE)
    directions = random.sample(list(DIRECTIONS), k=object_count)

    palette = list(range(10))
    bg = random.choice(palette)
    remaining = [color for color in palette if color != bg]
    random.shuffle(remaining)

    object_colors = remaining[:object_count]
    ray_colors = remaining[object_count : 2 * object_count]
    if len(ray_colors) < object_count:
        return None

    legend_order = list(range(object_count))
    random.shuffle(legend_order)
    legend_start = random.randint(0, width - object_count)

    inp = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)
    reserved_output = np.zeros((height, width), dtype=bool)
    object_mask = np.zeros((height, width), dtype=bool)

    for offset, idx in enumerate(legend_order):
        col = legend_start + offset
        inp[height - 2, col] = object_colors[idx]
        inp[height - 1, col] = ray_colors[idx]
        out[height - 2, col] = object_colors[idx]
        out[height - 1, col] = ray_colors[idx]
        reserved_output[height - 2, col] = True
        reserved_output[height - 1, col] = True

    specs = []
    for direction, obj_color, ray_color in zip(directions, object_colors, ray_colors, strict=False):
        spec = _make_shape(direction)
        spec["object_color"] = obj_color
        spec["ray_color"] = ray_color
        specs.append(spec)

    random.shuffle(specs)
    specs.sort(key=lambda spec: int(spec["area"]), reverse=True)

    for spec in specs:
        row_range, col_range = _position_ranges(spec, play_h, width)
        if not row_range or not col_range:
            return None

        candidates = [(r0, c0) for r0 in row_range for c0 in col_range]
        random.shuffle(candidates)
        placed = False

        for r0, c0 in candidates:
            coords = _object_cells(spec["mask"], r0, c0)
            ray = _ray_cells(spec, r0, c0)

            if _touches_existing_object(coords, object_mask):
                continue
            if not _coords_fit(coords, reserved_output):
                continue
            if any(reserved_output[r, c] for r, c in ray):
                continue

            _paint_cells(inp, coords, int(spec["object_color"]))
            _paint_cells(out, coords, int(spec["object_color"]))
            object_mask[coords[:, 0], coords[:, 1]] = True
            reserved_output[coords[:, 0], coords[:, 1]] = True

            for rr, cc in ray:
                out[rr, cc] = int(spec["ray_color"])
                reserved_output[rr, cc] = True

            placed = True
            break

        if not placed:
            return None

    _, legend_mask = _legend_mapping(inp, bg)
    if len(_find_components(inp, bg, legend_mask)) != object_count:
        return None

    if not np.array_equal(main(inp), out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
