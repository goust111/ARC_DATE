import numpy as np
from collections import deque


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def same_color_component(grid: np.ndarray, start: tuple[int, int], color: int) -> np.ndarray:
    q = deque([start])
    seen = {start}
    cells = []
    while q:
        r, c = q.popleft()
        cells.append((r, c))
        for nr, nc in neighbors4(r, c):
            if in_bounds(grid.shape, nr, nc) and (nr, nc) not in seen and int(grid[nr, nc]) == color:
                seen.add((nr, nc))
                q.append((nr, nc))
    return np.array(cells, dtype=int)


def normalize_shape(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def transform_shape(
    shape: tuple[tuple[int, int], ...], rot: int, flip_h: bool, flip_v: bool
) -> tuple[tuple[int, int], ...]:
    coords = list(shape)
    for _ in range(rot % 4):
        coords = [(c, -r) for r, c in coords]
    if flip_h:
        coords = [(r, -c) for r, c in coords]
    if flip_v:
        coords = [(-r, c) for r, c in coords]
    return normalize_shape(coords)


BASE_SHAPES = [
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 1), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
]


SHAPES = sorted(
    {
        transform_shape(shape, rot, flip_h, flip_v)
        for shape in BASE_SHAPES
        for rot in range(4)
        for flip_h in (False, True)
        for flip_v in (False, True)
    }
)


def place_shape(grid: np.ndarray, coords: tuple[tuple[int, int], ...], top: int, left: int, color: int) -> np.ndarray:
    pts = np.array([(top + r, left + c) for r, c in coords], dtype=int)
    grid[pts[:, 0], pts[:, 1]] = color
    return pts


def carrier_for_object(grid: np.ndarray, obj: np.ndarray) -> tuple[int, np.ndarray]:
    adjacent = []
    seen = set()
    for r, c in obj:
        for nr, nc in neighbors4(int(r), int(c)):
            if not in_bounds(grid.shape, nr, nc):
                continue
            color = int(grid[nr, nc])
            if color in (Color.BLACK, Color.RED):
                continue
            if (nr, nc) not in seen:
                seen.add((nr, nc))
                adjacent.append((color, (nr, nc)))
    assert adjacent, "red object must touch a carrier"
    carrier_color = adjacent[0][0]
    assert carrier_color in (Color.BLUE, Color.GREEN), "unexpected carrier color"
    assert all(color == carrier_color for color, _ in adjacent), "red object must belong to one carrier"
    carrier = same_color_component(grid, adjacent[0][1], carrier_color)
    return carrier_color, carrier


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each embedded red object horizontally to its carrier edge and mark the vacated cells in yellow.

    Input structure:
    - A black background contains 2-3 non-touching rectangular carrier regions colored blue or green.
    - Each carrier contains exactly one 4-connected red object of 3-5 cells, fully enclosed away from the carrier boundary.
    - Red objects are surrounded by their carrier color and different carriers stay separated by black background.

    Rule:
    - Identify each red object and the unique blue/green carrier region adjacent to it.
    - In a blue carrier, slide the whole red object right; in a green carrier, slide it left.
    - Move horizontally until the object's bounding box is flush with that carrier's boundary in the slide direction.
    - Cells from the original red position that are not covered by the moved object become yellow, while the moved object stays red and all remaining carrier/background cells stay unchanged.

    Key insight:
    - The carrier color chooses the horizontal slide direction, and the yellow trail is exactly the old red footprint minus the shifted red footprint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    red_objects = components_mask(g == Color.RED)
    assert red_objects, "expected at least one red object"

    out = g.copy()
    for obj in red_objects:
        carrier_color, carrier = carrier_for_object(g, obj)
        _, obj_c0, _, obj_c1 = bbox_coords(obj)
        _, car_c0, _, car_c1 = bbox_coords(carrier)
        width = obj_c1 - obj_c0 + 1
        if carrier_color == Color.BLUE:
            new_c0 = car_c1 - width + 1
        else:
            new_c0 = car_c0
        dc = new_c0 - obj_c0

        moved = obj.copy()
        moved[:, 1] += dc

        for r, c in obj:
            out[r, c] = carrier_color

        old_set = {tuple(map(int, cell)) for cell in obj}
        new_set = {tuple(map(int, cell)) for cell in moved}
        for cell in old_set - new_set:
            out[cell] = Color.YELLOW
        for r, c in new_set:
            out[r, c] = Color.RED

    return out


GEN_RETRIES = 500
GRID_MIN = 12
GRID_MAX = 19
CARRIER_COUNT_CHOICES = (2, 3)
RECT_HEIGHT_CHOICES = (4, 5, 6, 7)
RECT_WIDTH_CHOICES = (4, 5, 6, 7, 8, 9)
PLACEMENT_PAD = 1


def random_gap(
    rng: np.random.Generator,
    max_gap: int,
    shape_width: int,
    preferred: str | None = None,
) -> int:
    candidates = list(range(1, max_gap + 1))
    small = [gap for gap in candidates if gap < shape_width]
    large = [gap for gap in candidates if gap >= shape_width]
    if preferred == "overlap" and small:
        return int(rng.choice(small))
    if preferred == "disjoint" and large:
        return int(rng.choice(large))
    if small and large:
        pool = small if rng.random() < 0.5 else large
        return int(rng.choice(pool))
    return int(rng.choice(candidates))


def rects_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + PLACEMENT_PAD < br0
        or br1 + PLACEMENT_PAD < ar0
        or ac1 + PLACEMENT_PAD < bc0
        or bc1 + PLACEMENT_PAD < ac0
    )


def choose_shape(rng: np.random.Generator, carrier_h: int, carrier_w: int) -> tuple[tuple[int, int], ...] | None:
    choices = []
    for shape in SHAPES:
        rows = [r for r, _ in shape]
        cols = [c for _, c in shape]
        h = max(rows) + 1
        w = max(cols) + 1
        area = len(shape)
        if area < 3 or area > 5:
            continue
        if h <= carrier_h - 2 and w <= carrier_w - 2 and carrier_w - w - 1 >= 1:
            choices.append(shape)
    if not choices:
        return None
    # Prefer irregular shapes over straight lines but still keep some lines.
    irregular = [shape for shape in choices if len({r for r, _ in shape}) > 1 and len({c for _, c in shape}) > 1]
    if irregular and rng.random() < 0.75:
        return irregular[int(rng.integers(len(irregular)))]
    return choices[int(rng.integers(len(choices)))]


def gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX))
    w = int(rng.integers(GRID_MIN, GRID_MAX))
    grid = np.full((h, w), Color.BLACK, dtype=int)

    carrier_count = int(rng.choice(CARRIER_COUNT_CHOICES))
    colors = [Color.BLUE, Color.GREEN]
    if carrier_count == 3:
        colors.append(int(rng.choice([Color.BLUE, Color.GREEN])))
    rng.shuffle(colors)

    rects: list[tuple[int, int, int, int]] = []
    specs = []
    for color in colors:
        placed = False
        for _ in range(80):
            car_h = int(rng.choice(RECT_HEIGHT_CHOICES))
            car_w = int(rng.choice(RECT_WIDTH_CHOICES))
            shape = choose_shape(rng, car_h, car_w)
            if shape is None:
                continue
            if car_h >= h - 1 or car_w >= w - 1:
                continue
            r0 = int(rng.integers(1, h - car_h))
            c0 = int(rng.integers(1, w - car_w))
            rect = (r0, c0, r0 + car_h - 1, c0 + car_w - 1)
            if any(rects_too_close(rect, other) for other in rects):
                continue
            rects.append(rect)
            specs.append((rect, color, shape))
            placed = True
            break
        if not placed:
            return None

    branch_targets = ["overlap", "disjoint"]
    if carrier_count == 3:
        branch_targets.append("overlap" if rng.random() < 0.5 else "disjoint")

    has_overlap_case = False
    has_disjoint_case = False
    for idx, (rect, color, shape) in enumerate(specs):
        r0, c0, r1, c1 = rect
        car_h = r1 - r0 + 1
        shape_h = max(r for r, _ in shape) + 1
        shape_w = max(c for _, c in shape) + 1

        top_low = r0 + 1
        top_high = r1 - shape_h
        if top_low > top_high:
            return None
        top = int(rng.integers(top_low, top_high + 1))

        if color == Color.BLUE:
            final_left = c1 - shape_w + 1
            max_gap = final_left - (c0 + 1)
            if max_gap < 1:
                return None
            gap = random_gap(rng, max_gap, shape_w, branch_targets[idx])
            start_left = final_left - gap
        else:
            final_left = c0
            max_gap = (c1 - 1) - (final_left + shape_w - 1)
            if max_gap < 1:
                return None
            gap = random_gap(rng, max_gap, shape_w, branch_targets[idx])
            start_left = final_left + gap

        grid[r0 : r1 + 1, c0 : c1 + 1] = color
        pts = place_shape(grid, shape, top, start_left, Color.RED)

        if gap < shape_w:
            has_overlap_case = True
        else:
            has_disjoint_case = True

        # Ensure the object stays fully enclosed in the input.
        obj_r0, obj_c0, obj_r1, obj_c1 = bbox_coords(pts)
        if not (r0 + 1 <= obj_r0 and obj_r1 <= r1 - 1 and c0 + 1 <= obj_c0 and obj_c1 <= c1 - 1):
            return None

    if not (Color.BLUE in colors and Color.GREEN in colors):
        return None
    if not (has_overlap_case and has_disjoint_case):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if not np.any(out == Color.YELLOW):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
