import numpy as np


BLUE = 1
FG_COLORS = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_objects(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLUE or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )

    return objects


def _touching_side(obj: dict[str, object], shape: tuple[int, int]) -> str:
    coords = obj["coords"]
    h, w = shape
    touches: list[str] = []
    if np.any(coords[:, 0] == 0):
        touches.append("top")
    if np.any(coords[:, 0] == h - 1):
        touches.append("bottom")
    if np.any(coords[:, 1] == 0):
        touches.append("left")
    if np.any(coords[:, 1] == w - 1):
        touches.append("right")
    assert len(touches) == 1, "each object must touch exactly one grid edge"
    return touches[0]


def _object_mask(obj: dict[str, object]) -> np.ndarray:
    r0, c0, r1, c1 = obj["bbox"]
    coords = obj["coords"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local = coords - np.array([r0, c0], dtype=int)
    mask[local[:, 0], local[:, 1]] = True
    return mask


def _flip_mask(mask: np.ndarray, side: str) -> np.ndarray:
    if side in ("top", "bottom"):
        return np.fliplr(mask)
    return np.flipud(mask)


def _transform_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for obj in _find_objects(g):
        side = _touching_side(obj, g.shape)
        color = int(obj["color"])
        r0, c0, r1, c1 = obj["bbox"]
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BLUE, dtype=int)
        patch[_object_mask(obj)] = color
        if side in ("top", "bottom"):
            out[r0 : r1 + 1, c0 : c1 + 1] = np.fliplr(patch)
        else:
            out[r0 : r1 + 1, c0 : c1 + 1] = np.flipud(patch)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror each edge-touching foreground object inside its own bounding box.

    Input structure:
    - The background is blue.
    - The non-blue pixels form several disjoint monochromatic 4-connected objects.
    - Every object touches exactly one outer grid edge: top, bottom, left, or right.

    Rule:
    - Find each foreground object and its tight bounding box.
    - If the object touches the top or bottom edge, flip its bbox contents left-to-right.
    - If the object touches the left or right edge, flip its bbox contents top-to-bottom.
    - Put every mirrored bbox back in the same location and leave the blue background unchanged.

    Key insight:
    - The mirror axis is orthogonal to the contacted edge: horizontal-edge objects use a vertical mirror axis, and vertical-edge objects use a horizontal mirror axis.
    """
    return _transform_grid(grid)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _edge_positions(height: int, width: int, side: str) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(width)]
    if side == "bottom":
        return [(height - 1, c) for c in range(width)]
    if side == "left":
        return [(r, 0) for r in range(height)]
    return [(r, width - 1) for r in range(height)]


def _grow_mask(rng: np.random.Generator, height: int, width: int, side: str) -> np.ndarray | None:
    max_area = height * width - 1
    if max_area < 3:
        return None

    target_area = int(rng.integers(max(3, (height * width + 2) // 3), max_area + 1))
    mask = np.zeros((height, width), dtype=bool)
    edge = _edge_positions(height, width, side)
    start_r, start_c = edge[int(rng.integers(len(edge)))]
    mask[start_r, start_c] = True

    while int(mask.sum()) < target_area:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break
        nr, nc = list(frontier)[int(rng.integers(len(frontier)))]
        mask[nr, nc] = True

    return _trim_mask(mask)


def _make_object_mask(rng: np.random.Generator, side: str) -> np.ndarray | None:
    for _ in range(200):
        if side in ("top", "bottom"):
            height = int(rng.integers(2, 6))
            width = int(rng.integers(3, 7))
        else:
            height = int(rng.integers(3, 7))
            width = int(rng.integers(2, 6))

        mask = _grow_mask(rng, height, width, side)
        if mask is None:
            continue
        if side in ("top", "bottom") and mask.shape[1] < 2:
            continue
        if side in ("left", "right") and mask.shape[0] < 2:
            continue
        if np.array_equal(mask, _flip_mask(mask, side)):
            continue
        return mask
    return None


def _bboxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    separated = ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0
    return not separated


def _candidate_positions(
    height: int, width: int, side: str, box_h: int, box_w: int
) -> list[tuple[int, int]]:
    if side in ("top", "bottom"):
        if width - box_w < 2 or height - box_h < 1:
            return []
        row = 0 if side == "top" else height - box_h
        return [(row, col) for col in range(1, width - box_w)]
    if height - box_h < 2 or width - box_w < 1:
        return []
    col = 0 if side == "left" else width - box_w
    return [(row, col) for row in range(1, height - box_h)]


def _place_bbox(
    rng: np.random.Generator,
    used_boxes: list[tuple[int, int, int, int]],
    height: int,
    width: int,
    side: str,
    box_h: int,
    box_w: int,
) -> tuple[int, int, int, int] | None:
    starts = _candidate_positions(height, width, side, box_h, box_w)
    rng.shuffle(starts)
    for r0, c0 in starts:
        bbox = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        if all(not _bboxes_too_close(bbox, other) for other in used_boxes):
            return bbox
    return None


def _coords_too_close(a: np.ndarray, b: np.ndarray) -> bool:
    for r1, c1 in a:
        diffs = np.abs(b - np.array([r1, c1], dtype=int))
        if np.any((diffs[:, 0] <= 1) & (diffs[:, 1] <= 1)):
            return True
    return False


def _validate_generated_grid(grid: np.ndarray, expected_count: int) -> bool:
    try:
        objects = _find_objects(grid)
        if len(objects) != expected_count:
            return False

        seen_colors: set[int] = set()
        seen_sides: set[str] = set()
        for obj in objects:
            color = int(obj["color"])
            if color == BLUE or color in seen_colors:
                return False
            seen_colors.add(color)

            side = _touching_side(obj, grid.shape)
            seen_sides.add(side)
            mask = _object_mask(obj)
            if np.array_equal(mask, _flip_mask(mask, side)):
                return False

        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                if _coords_too_close(objects[i]["coords"], objects[j]["coords"]):
                    return False

        if seen_sides != set(SIDES):
            return False

        transformed = main(grid)
        if np.array_equal(transformed, grid):
            return False
        if not np.array_equal(main(transformed), grid):
            return False
    except AssertionError:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(16, 29))
    width = int(rng.integers(16, 29))
    n_objects = int(rng.integers(4, 7))

    sides = list(SIDES)
    if n_objects > 4:
        sides.extend(rng.choice(np.array(SIDES), size=n_objects - 4, replace=True).tolist())
    rng.shuffle(sides)

    colors = rng.choice(FG_COLORS, size=n_objects, replace=False)
    grid = np.full((height, width), BLUE, dtype=int)
    used_boxes: list[tuple[int, int, int, int]] = []

    for side, color in zip(sides, colors):
        placed = False
        for _ in range(120):
            mask = _make_object_mask(rng, side)
            if mask is None:
                continue
            bbox = _place_bbox(rng, used_boxes, height, width, side, mask.shape[0], mask.shape[1])
            if bbox is None:
                continue

            r0, c0, r1, c1 = bbox
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if np.any(patch[mask] != BLUE):
                continue
            patch[mask] = int(color)
            used_boxes.append(bbox)
            placed = True
            break

        if not placed:
            return None

    if not _validate_generated_grid(grid, n_objects):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
