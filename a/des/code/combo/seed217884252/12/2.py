import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG_COLOR = 7
NON_BG_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]

GRID_MIN = 24
GRID_MAX = 30
OBJECT_COUNT_OPTIONS = (2, 3)
SHAPE_RETRIES = 200
PLACEMENT_RETRIES = 400
GEN_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if bool(mask[r, c]) or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches_edge = False
            while stack:
                rr, cc = stack.pop()
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_edge = True
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]) and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def _component_holes(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return _count_holes(mask)


def _segment_start(center: int, length: int) -> int:
    return int(center) - (int(length) - 1) // 2


def _breakpoint_indices(length: int, hole_count: int) -> list[int]:
    if hole_count <= 0:
        return []
    assert length > hole_count, "line length must exceed the number of breakpoints"
    out: list[int] = []
    prev = -1
    for k in range(1, hole_count + 1):
        raw = k * (length + 1) / (hole_count + 1)
        pos = int(np.floor(raw + 0.5)) - 1
        min_pos = prev + 1
        max_pos = length - (hole_count - k) - 1
        pos = max(min_pos, min(pos, max_pos))
        out.append(pos)
        prev = pos
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each 4-connected foreground object to a centered 1D segment whose
    length equals the object's area, then erase equally spaced breakpoint cells
    according to the object's enclosed-hole count.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)

    comps = _find_components(g, bg)
    if not comps:
        return g.copy()

    out = np.full((h, w), int(bg), dtype=int)
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        length = int(comp["area"])
        hole_count = _component_holes(comp["coords"], comp["bbox"])
        color = int(comp["color"])

        if box_w >= box_h:
            row = (r0 + r1) // 2
            center = (c0 + c1) // 2
            start = _segment_start(center, length)
            assert 0 <= start and start + length <= w
            out[row, start : start + length] = color
            for idx in _breakpoint_indices(length, hole_count):
                out[row, start + idx] = bg
        else:
            col = (c0 + c1) // 2
            center = (r0 + r1) // 2
            start = _segment_start(center, length)
            assert 0 <= start and start + length <= h
            out[start : start + length, col] = color
            for idx in _breakpoint_indices(length, hole_count):
                out[start + idx, col] = bg

    return out


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _sample_hole_cells(height: int, width: int, hole_count: int) -> list[tuple[int, int]] | None:
    if hole_count == 0:
        return []
    interior = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    if len(interior) < hole_count:
        return None
    for _ in range(80):
        cells = interior[:]
        random.shuffle(cells)
        chosen: list[tuple[int, int]] = []
        for cell in cells:
            if all(max(abs(cell[0] - rr), abs(cell[1] - cc)) > 1 for rr, cc in chosen):
                chosen.append(cell)
                if len(chosen) == hole_count:
                    return chosen
    return None


def _random_refine(mask: np.ndarray, hole_count: int) -> np.ndarray:
    out = mask.copy()
    max_removals = min(6, int(out.sum()) - max(6, hole_count + 4))
    if max_removals <= 0:
        return out
    target = random.randint(0, max_removals)
    attempts = 0
    removed = 0

    while removed < target and attempts < 200:
        attempts += 1
        coords = np.argwhere(out)
        rr, cc = map(int, coords[random.randrange(len(coords))])
        trial = out.copy()
        trial[rr, cc] = False
        if not _touches_all_sides(trial):
            continue
        if not _mask_connected(trial):
            continue
        if _count_holes(trial) != hole_count:
            continue
        out = trial
        removed += 1
    return out


def _shape_choices(horizontal: bool) -> list[tuple[int, int]]:
    choices = []
    for height in range(3, 7):
        for width in range(4, 8):
            if horizontal and width < height:
                continue
            if not horizontal and height <= width:
                continue
            choices.append((height, width))
    return choices


def _generate_shape(horizontal: bool, hole_count: int) -> np.ndarray | None:
    choices = _shape_choices(horizontal)
    for _ in range(SHAPE_RETRIES):
        height, width = random.choice(choices)
        holes = _sample_hole_cells(height, width, hole_count)
        if holes is None:
            continue

        mask = np.ones((height, width), dtype=bool)
        for rr, cc in holes:
            mask[rr, cc] = False
        mask = _random_refine(mask, hole_count)

        if not _touches_all_sides(mask):
            continue
        if not _mask_connected(mask):
            continue
        if _count_holes(mask) != hole_count:
            continue

        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if int(mask.sum()) <= hole_count + 1:
            continue
        return mask
    return None


def _segment_cells_for_bbox(
    shape: np.ndarray,
    top: int,
    left: int,
    grid_h: int,
    grid_w: int,
) -> list[tuple[int, int]] | None:
    height, width = shape.shape
    area = int(shape.sum())
    if width >= height:
        row = top + (height - 1) // 2
        center = left + (width - 1) // 2
        start = _segment_start(center, area)
        if start < 0 or start + area > grid_w:
            return None
        return [(row, start + i) for i in range(area)]

    col = left + (width - 1) // 2
    center = top + (height - 1) // 2
    start = _segment_start(center, area)
    if start < 0 or start + area > grid_h:
        return None
    return [(start + i, col) for i in range(area)]


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return not (
        a[2] < b[0] - gap
        or b[2] < a[0] - gap
        or a[3] < b[1] - gap
        or b[3] < a[1] - gap
    )


def _paint_shape(grid: np.ndarray, shape: np.ndarray, top: int, left: int, color: int) -> None:
    coords = np.argwhere(shape)
    for rr, cc in coords:
        grid[top + int(rr), left + int(cc)] = int(color)


def _candidate_ranges(shape: np.ndarray, grid_h: int, grid_w: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    height, width = shape.shape
    area = int(shape.sum())
    if width >= height:
        center_offset = (width - 1) // 2
        c_low = max(0, (area - 1) // 2 - center_offset)
        c_high = min(grid_w - width, grid_w - 1 - area // 2 - center_offset)
        if c_low > c_high:
            return None
        return (0, grid_h - height), (c_low, c_high)

    center_offset = (height - 1) // 2
    r_low = max(0, (area - 1) // 2 - center_offset)
    r_high = min(grid_h - height, grid_h - 1 - area // 2 - center_offset)
    if r_low > r_high:
        return None
    return (r_low, r_high), (0, grid_w - width)


def _validate_generated_input(grid: np.ndarray) -> bool:
    bg = _background_color(grid)
    comps = _find_components(grid, bg)
    if len(comps) < 2:
        return False
    full_segment_cells: set[tuple[int, int]] = set()
    seen_colors = set()
    for comp in comps:
        if int(comp["color"]) in seen_colors:
            return False
        seen_colors.add(int(comp["color"]))
        length = int(comp["area"])
        r0, c0, r1, c1 = comp["bbox"]
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        if box_w >= box_h:
            row = (r0 + r1) // 2
            center = (c0 + c1) // 2
            start = _segment_start(center, length)
            if start < 0 or start + length > grid.shape[1]:
                return False
            cells = {(row, start + i) for i in range(length)}
        else:
            col = (c0 + c1) // 2
            center = (r0 + r1) // 2
            start = _segment_start(center, length)
            if start < 0 or start + length > grid.shape[0]:
                return False
            cells = {(start + i, col) for i in range(length)}
        if full_segment_cells & cells:
            return False
        full_segment_cells |= cells
        if _component_holes(comp["coords"], comp["bbox"]) > 3:
            return False
    return np.count_nonzero(grid == bg) > grid.size // 2


def _gen_attempt() -> np.ndarray | None:
    object_count = random.choice(OBJECT_COUNT_OPTIONS)
    horizontal_count = random.randint(1, object_count - 1)
    orientations = [True] * horizontal_count + [False] * (object_count - horizontal_count)
    random.shuffle(orientations)

    hole_counts = random.sample([0, 1, 2, 3], k=object_count)
    colors = random.sample(NON_BG_COLORS, k=object_count)

    shapes = []
    for horizontal, hole_count, color in zip(orientations, hole_counts, colors):
        shape = _generate_shape(horizontal, hole_count)
        if shape is None:
            return None
        shapes.append(
            {
                "shape": shape,
                "horizontal": horizontal,
                "hole_count": hole_count,
                "color": color,
                "area": int(shape.sum()),
            }
        )

    min_h = max(GRID_MIN, max((obj["area"] if not obj["horizontal"] else 0) for obj in shapes) + 4)
    min_w = max(GRID_MIN, max((obj["area"] if obj["horizontal"] else 0) for obj in shapes) + 4)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None
    grid_h = random.randint(min_h, GRID_MAX)
    grid_w = random.randint(min_w, GRID_MAX)

    order = sorted(range(object_count), key=lambda i: (-shapes[i]["area"], shapes[i]["horizontal"]))
    placed_boxes: list[tuple[int, int, int, int]] = []
    used_segment_cells: set[tuple[int, int]] = set()
    placements: dict[int, tuple[int, int]] = {}

    for idx in order:
        obj = shapes[idx]
        shape = obj["shape"]
        ranges = _candidate_ranges(shape, grid_h, grid_w)
        if ranges is None:
            return None
        (r_low, r_high), (c_low, c_high) = ranges
        if r_low > r_high or c_low > c_high:
            return None

        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(r_low, r_high)
            left = random.randint(c_low, c_high)
            box = (top, left, top + shape.shape[0] - 1, left + shape.shape[1] - 1)
            if any(_boxes_too_close(box, other, gap=1) for other in placed_boxes):
                continue
            seg = _segment_cells_for_bbox(shape, top, left, grid_h, grid_w)
            if seg is None or any(cell in used_segment_cells for cell in seg):
                continue
            placements[idx] = (top, left)
            placed_boxes.append(box)
            used_segment_cells.update(seg)
            break
        else:
            return None

    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    for idx, obj in enumerate(shapes):
        top, left = placements[idx]
        _paint_shape(grid, obj["shape"], top, left, obj["color"])

    if not _validate_generated_input(grid):
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
