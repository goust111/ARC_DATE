import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

MAX_GENERATION_ATTEMPTS = 400


def _find_components(grid: np.ndarray, bg: int = BLACK) -> list[np.ndarray]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))

    components.sort(
        key=lambda coords: (
            int(coords[:, 0].min()),
            int(coords[:, 1].min()),
            int(coords[:, 0].max()),
            int(coords[:, 1].max()),
        )
    )
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _outline_color(height: int, width: int) -> int:
    if width > height:
        return BLUE
    if width < height:
        return RED
    return GREEN


def _draw_outline(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each foreground object with a color-coded outline of its bounding box.

    Horizontal boxes become blue, vertical boxes become red, and square boxes
    become green. The original object is erased.
    """
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BLACK)

    for coords in _find_components(g, bg=BLACK):
        r0, c0, r1, c1 = _bbox(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        _draw_outline(out, (r0, c0, r1, c1), _outline_color(height, width))

    return out


def _manhattan_path(
    start: tuple[int, int], end: tuple[int, int], row_first: bool
) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = end
    path = [(r, c)]

    def step_rows() -> None:
        nonlocal r
        while r != tr:
            r += 1 if tr > r else -1
            path.append((r, c))

    def step_cols() -> None:
        nonlocal c
        while c != tc:
            c += 1 if tc > c else -1
            path.append((r, c))

    if row_first:
        step_rows()
        step_cols()
    else:
        step_cols()
        step_rows()
    return path


def _random_object_patch(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    center = (int(rng.integers(0, height)), int(rng.integers(0, width)))
    anchors = [
        (0, int(rng.integers(0, width))),
        (height - 1, int(rng.integers(0, width))),
        (int(rng.integers(0, height)), 0),
        (int(rng.integers(0, height)), width - 1),
    ]

    cells = {center}
    for anchor in anchors:
        for cell in _manhattan_path(center, anchor, row_first=bool(rng.integers(0, 2))):
            cells.add(cell)

    area = height * width
    min_fill = max(len(cells), int(np.ceil(area * 0.35)))
    max_fill = max(min_fill, int(np.ceil(area * 0.75)))
    target = int(rng.integers(min_fill, max_fill + 1))

    while len(cells) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(frontier[int(rng.integers(0, len(frontier)))])

    patch = np.full((height, width), BLACK, dtype=int)
    for r, c in cells:
        patch[r, c] = GREY
    return patch


def _sample_dims(category: str, rng: np.random.Generator) -> tuple[int, int]:
    if category == "horizontal":
        height = int(rng.integers(2, 7))
        width = int(rng.integers(height + 1, min(10, height + 5) + 1))
        return height, width
    if category == "vertical":
        width = int(rng.integers(2, 7))
        height = int(rng.integers(width + 1, min(10, width + 5) + 1))
        return height, width
    side = int(rng.integers(2, 8))
    return side, side


def _reserve_box(
    occupied: np.ndarray, r0: int, c0: int, height: int, width: int, padding: int
) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(h, r0 + height + padding)
    cc1 = min(w, c0 + width + padding)
    if occupied[rr0:rr1, cc0:cc1].any():
        return False
    occupied[rr0:rr1, cc0:cc1] = True
    return True


def _place_objects(
    grid_shape: tuple[int, int], objects: list[dict], rng: np.random.Generator
) -> list[dict] | None:
    height, width = grid_shape
    occupied = np.zeros((height, width), dtype=bool)
    placed: list[dict] = []

    order = sorted(
        range(len(objects)),
        key=lambda idx: objects[idx]["patch"].shape[0] * objects[idx]["patch"].shape[1],
        reverse=True,
    )

    for idx in order:
        patch = objects[idx]["patch"]
        box_h, box_w = patch.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(height - box_h + 1):
            for c0 in range(width - box_w + 1):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(height, r0 + box_h + 1)
                cc1 = min(width, c0 + box_w + 1)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = candidates[int(rng.integers(0, len(candidates)))]
        ok = _reserve_box(occupied, r0, c0, box_h, box_w, padding=1)
        if not ok:
            return None
        placed.append(
            {
                "category": objects[idx]["category"],
                "patch": patch,
                "r0": r0,
                "c0": c0,
            }
        )

    return placed


def _expected_output(grid_shape: tuple[int, int], placed: list[dict]) -> np.ndarray:
    out = np.full(grid_shape, BLACK, dtype=int)
    for obj in placed:
        patch = obj["patch"]
        box_h, box_w = patch.shape
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        color = _outline_color(box_h, box_w)
        _draw_outline(out, (r0, c0, r0 + box_h - 1, c0 + box_w - 1), color)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    categories = ["horizontal", "vertical", "square"]
    extra_count = int(rng.integers(0, 4))
    extras = list(rng.choice(categories, size=extra_count, replace=True))
    categories.extend(str(cat) for cat in extras)

    objects = []
    for category in categories:
        box_h, box_w = _sample_dims(category, rng)
        objects.append(
            {
                "category": category,
                "patch": _random_object_patch(box_h, box_w, rng),
            }
        )

    total_box_area = sum(obj["patch"].shape[0] * obj["patch"].shape[1] for obj in objects)
    max_box_h = max(obj["patch"].shape[0] for obj in objects)
    max_box_w = max(obj["patch"].shape[1] for obj in objects)
    base_side = int(np.ceil(np.sqrt(total_box_area * 3.5)))
    grid_h = int(rng.integers(max(16, max_box_h + 6, base_side), max(17, max_box_h + 13, base_side + 8)))
    grid_w = int(rng.integers(max(16, max_box_w + 6, base_side), max(17, max_box_w + 13, base_side + 8)))

    placed = _place_objects((grid_h, grid_w), objects, rng)
    if placed is None:
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    for obj in placed:
        patch = obj["patch"]
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        mask = patch == GREY
        grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = GREY

    expected = _expected_output(grid.shape, placed)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
