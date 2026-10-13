import numpy as np
from collections import deque


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()


GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 160
SHAPE_RETRIES = 80

MIN_OBJECTS = 2
MAX_OBJECTS = 4
MIN_GRID_SIDE = 14
MAX_GRID_SIDE = 28

SHORT_AXIS_CHOICES = (3, 5, 7)
LONG_EXTRA_MIN = 2
LONG_EXTRA_MAX = 7


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _component_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress every foreground object onto the center line of its longest bbox axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    out = np.full_like(g, bg)

    for comp in find_components(g, bg):
        r0, c0, r1, c1 = comp["bbox"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        color = int(comp["color"])

        if width > height:
            row = (r0 + r1) // 2
            out[row, c0 : c1 + 1] = color
        else:
            col = (c0 + c1) // 2
            out[r0 : r1 + 1, col] = color

    return out


def _has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    empty = ~mask
    seen = np.zeros_like(mask, dtype=bool)
    queue = deque()

    for r in range(h):
        for c in (0, w - 1):
            if empty[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if empty[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))

    return bool(np.any(empty & ~seen))


def _shrink_segment(start: int, end: int, min_len: int) -> tuple[int, int]:
    length = end - start + 1
    max_trim = max(0, length - min_len)
    if max_trim == 0:
        return start, end

    step_cap = min(2, max_trim)
    trim_left = int(RNG.integers(0, step_cap + 1))
    trim_right = int(RNG.integers(0, min(step_cap, max_trim - trim_left) + 1))
    if trim_left == 0 and trim_right == 0:
        if RNG.random() < 0.5:
            trim_left = 1
        else:
            trim_right = 1
    return start + trim_left, end - trim_right


def _make_horizontal_mask(height: int, width: int) -> np.ndarray | None:
    intervals = [None] * height
    anchor = int(RNG.integers(0, height))
    intervals[anchor] = (0, width - 1)
    min_len = max(2, width // 3)

    start, end = 0, width - 1
    for r in range(anchor - 1, -1, -1):
        start, end = _shrink_segment(start, end, min_len)
        intervals[r] = (start, end)

    start, end = 0, width - 1
    for r in range(anchor + 1, height):
        start, end = _shrink_segment(start, end, min_len)
        intervals[r] = (start, end)

    mask = np.zeros((height, width), dtype=bool)
    for r, (lo, hi) in enumerate(intervals):
        mask[r, lo : hi + 1] = True

    if all(lo == 0 and hi == width - 1 for lo, hi in intervals):
        return None
    return mask


def _make_vertical_mask(height: int, width: int) -> np.ndarray | None:
    intervals = [None] * width
    anchor = int(RNG.integers(0, width))
    intervals[anchor] = (0, height - 1)
    min_len = max(2, height // 3)

    start, end = 0, height - 1
    for c in range(anchor - 1, -1, -1):
        start, end = _shrink_segment(start, end, min_len)
        intervals[c] = (start, end)

    start, end = 0, height - 1
    for c in range(anchor + 1, width):
        start, end = _shrink_segment(start, end, min_len)
        intervals[c] = (start, end)

    mask = np.zeros((height, width), dtype=bool)
    for c, (lo, hi) in enumerate(intervals):
        mask[lo : hi + 1, c] = True

    if all(lo == 0 and hi == height - 1 for lo, hi in intervals):
        return None
    return mask


def _generate_object(orientation: str) -> np.ndarray | None:
    short_axis = int(RNG.choice(SHORT_AXIS_CHOICES))
    long_axis = int(short_axis + RNG.integers(LONG_EXTRA_MIN, LONG_EXTRA_MAX + 1))

    if orientation == "h":
        height, width = short_axis, long_axis
        builder = _make_horizontal_mask
    else:
        height, width = long_axis, short_axis
        builder = _make_vertical_mask

    for _ in range(SHAPE_RETRIES):
        mask = builder(height, width)
        if mask is None:
            continue
        if _has_hole(mask):
            continue
        if np.count_nonzero(mask) == mask.size:
            continue
        return mask
    return None


def _orientation_list(num_objects: int) -> list[str]:
    horiz = num_objects // 2
    vert = num_objects // 2
    if num_objects % 2 == 1:
        if RNG.random() < 0.5:
            horiz += 1
        else:
            vert += 1
    orientations = ["h"] * horiz + ["v"] * vert
    RNG.shuffle(orientations)
    return orientations


def _place_objects(grid: np.ndarray, objects: list[dict]) -> bool:
    h, w = grid.shape
    reserved = np.zeros((h, w), dtype=bool)

    for obj in sorted(objects, key=lambda item: item["mask"].size, reverse=True):
        mask = obj["mask"]
        oh, ow = mask.shape
        coords = np.argwhere(mask)
        placed = False

        for _ in range(PLACEMENT_RETRIES):
            r0 = int(RNG.integers(0, h - oh + 1))
            c0 = int(RNG.integers(0, w - ow + 1))
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + oh + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + ow + 1)
            if np.any(reserved[rr0:rr1, cc0:cc1]):
                continue

            pixels = coords + np.array([r0, c0], dtype=int)
            grid[pixels[:, 0], pixels[:, 1]] = obj["color"]
            reserved[rr0:rr1, cc0:cc1] = True
            placed = True
            break

        if not placed:
            return False

    return True


def _valid_generated_grid(grid: np.ndarray) -> bool:
    bg = get_bg(grid)
    if np.count_nonzero(grid == bg) <= grid.size // 2:
        return False

    comps = find_components(grid, bg)
    if len(comps) < 2:
        return False

    axes = set()
    colors = set()
    for comp in comps:
        color = int(comp["color"])
        if color in colors:
            return False
        colors.add(color)

        r0, c0, r1, c1 = comp["bbox"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height == width:
            return False
        axes.add("h" if width > height else "v")

        mask = _component_mask(comp["coords"], comp["bbox"])
        if _has_hole(mask):
            return False

    if axes != {"h", "v"}:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    return not np.array_equal(out, grid)


def _gen_attempt() -> np.ndarray | None:
    num_objects = int(RNG.integers(MIN_OBJECTS, MAX_OBJECTS + 1))
    base_side = max(MIN_GRID_SIDE, 8 + 3 * num_objects)
    height = int(RNG.integers(base_side, MAX_GRID_SIDE + 1))
    width = int(RNG.integers(base_side, MAX_GRID_SIDE + 1))

    bg = int(RNG.integers(0, 10))
    palette = [c for c in range(10) if c != bg]
    RNG.shuffle(palette)

    orientations = _orientation_list(num_objects)
    objects = []
    for idx, orientation in enumerate(orientations):
        mask = _generate_object(orientation)
        if mask is None:
            return None
        objects.append({"mask": mask, "color": int(palette[idx]), "orientation": orientation})

    grid = np.full((height, width), bg, dtype=int)
    if not _place_objects(grid, objects):
        return None
    if not _valid_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
