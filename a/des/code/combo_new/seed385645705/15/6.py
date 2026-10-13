import numpy as np


GEN_BG = 0
GEN_CONTAINER = 3
PAYLOAD_COLORS = np.array([1, 2, 4, 5, 6, 7, 8, 9], dtype=int)
DIRECTIONS = ("up", "down", "left", "right")
GENERATE_RETRIES = 400
PLACE_RETRIES = 120
GRID_MAX = 30
RESERVE_PAD = 1
CONTAINER_MIN = 4
CONTAINER_MAX = 7


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _cc4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                if rr > 0 and mask[rr - 1, cc] and not seen[rr - 1, cc]:
                    seen[rr - 1, cc] = True
                    stack.append((rr - 1, cc))
                if rr + 1 < h and mask[rr + 1, cc] and not seen[rr + 1, cc]:
                    seen[rr + 1, cc] = True
                    stack.append((rr + 1, cc))
                if cc > 0 and mask[rr, cc - 1] and not seen[rr, cc - 1]:
                    seen[rr, cc - 1] = True
                    stack.append((rr, cc - 1))
                if cc + 1 < w and mask[rr, cc + 1] and not seen[rr, cc + 1]:
                    seen[rr, cc + 1] = True
                    stack.append((rr, cc + 1))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _container_template(height: int, width: int, open_dir: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    if open_dir == "up":
        mask[0, 1:-1] = False
    elif open_dir == "down":
        mask[-1, 1:-1] = False
    elif open_dir == "left":
        mask[1:-1, 0] = False
    elif open_dir == "right":
        mask[1:-1, -1] = False
    else:
        raise ValueError(open_dir)
    return mask


def _match_container(local_mask: np.ndarray) -> str | None:
    height, width = local_mask.shape
    if height < 3 or width < 3:
        return None
    matches = [
        open_dir
        for open_dir in DIRECTIONS
        if np.array_equal(local_mask, _container_template(height, width, open_dir))
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _container_candidates(grid: np.ndarray, bg: int) -> dict[int, list[dict]]:
    candidates: dict[int, list[dict]] = {}
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _cc4(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local[coords[:, 0] - r0, coords[:, 1] - c0] = True
            open_dir = _match_container(local)
            if open_dir is None:
                continue
            candidates.setdefault(color, []).append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "open_dir": open_dir,
                }
            )
    return candidates


def _find_containers(grid: np.ndarray, bg: int) -> list[dict]:
    candidates = _container_candidates(grid, bg)
    if not candidates:
        return []
    best_color = max(
        candidates,
        key=lambda color: (
            len(candidates[color]),
            sum(int(comp["coords"].shape[0]) for comp in candidates[color]),
        ),
    )
    containers = candidates[best_color]
    containers.sort(key=lambda comp: comp["bbox"])
    return containers


def _payload_cells(
    grid: np.ndarray, bbox: tuple[int, int, int, int], wall_color: int, bg: int
) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = bbox
    payload = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            color = int(grid[r, c])
            if color == bg or color == wall_color:
                continue
            payload.append((r, c, color))
    return payload


def _shift_payload(
    payload: list[tuple[int, int, int]],
    bbox: tuple[int, int, int, int],
    open_dir: str,
) -> list[tuple[int, int, int]]:
    if not payload:
        return []
    r0, c0, r1, c1 = bbox
    rows = np.array([r for r, _, _ in payload], dtype=int)
    cols = np.array([c for _, c, _ in payload], dtype=int)
    if open_dir == "up":
        dr, dc = r0 - int(rows.max()), 0
    elif open_dir == "down":
        dr, dc = r1 - int(rows.min()), 0
    elif open_dir == "left":
        dr, dc = 0, c0 - int(cols.max())
    elif open_dir == "right":
        dr, dc = 0, c1 - int(cols.min())
    else:
        raise ValueError(open_dir)
    return [(r + dr, c + dc, color) for r, c, color in payload]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pour each container's interior colored pixels out through its opening while preserving their relative arrangement.

    Input structure:
    - The background is black or otherwise the dominant color.
    - One structural color forms several 4-connected three-sided rectangular containers with a one-cell-thick wall.
    - Each container may contain scattered non-background, non-wall colored pixels strictly inside its bbox, and the open side faces up, down, left, or right.

    Rule:
    - Detect every three-sided rectangular container and identify which side is missing from its border.
    - Collect all colored payload pixels inside that container and translate them together in the opening direction until the whole payload sits just outside the container.
    - Preserve all relative offsets inside the payload, clear the original payload cells to background, and leave the container walls unchanged.
    - Containers whose interiors contain no payload pixels remain unchanged.

    Key insight:
    - The transformation is controlled purely by container geometry: the missing border side encodes the ejection direction, and every interior payload cell follows the same rigid translation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    containers = _find_containers(g, bg)
    if not containers:
        return g.copy()

    out = g.copy()
    moved_cells: list[tuple[int, int, int]] = []
    h, w = g.shape

    for container in containers:
        payload = _payload_cells(g, container["bbox"], int(container["color"]), bg)
        if not payload:
            continue
        for r, c, _ in payload:
            out[r, c] = bg
        moved_cells.extend(_shift_payload(payload, container["bbox"], container["open_dir"]))

    for r, c, color in moved_cells:
        assert 0 <= r < h and 0 <= c < w
        out[r, c] = color

    return out


def _reserve_box(
    r0: int, c0: int, height: int, width: int, open_dir: str
) -> tuple[int, int, int, int]:
    top = r0
    left = c0
    bottom = r0 + height - 1
    right = c0 + width - 1
    if open_dir == "up":
        top -= height - 2
    elif open_dir == "down":
        bottom += height - 2
    elif open_dir == "left":
        left -= width - 2
    elif open_dir == "right":
        right += width - 2
    else:
        raise ValueError(open_dir)
    return (
        max(0, top - RESERVE_PAD),
        max(0, left - RESERVE_PAD),
        min(GRID_MAX - 1, bottom + RESERVE_PAD),
        min(GRID_MAX - 1, right + RESERVE_PAD),
    )


def _placement_ranges(
    grid_h: int, grid_w: int, height: int, width: int, open_dir: str
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    extra_top = height - 2 if open_dir == "up" else 0
    extra_bottom = height - 2 if open_dir == "down" else 0
    extra_left = width - 2 if open_dir == "left" else 0
    extra_right = width - 2 if open_dir == "right" else 0

    r_min = RESERVE_PAD + extra_top
    r_max = grid_h - height - RESERVE_PAD - extra_bottom
    c_min = RESERVE_PAD + extra_left
    c_max = grid_w - width - RESERVE_PAD - extra_right

    if r_min > r_max or c_min > c_max:
        return None
    return (r_min, r_max), (c_min, c_max)


def _empty_flags(count: int, rng: np.random.Generator) -> np.ndarray:
    if count == 1:
        return np.array([False], dtype=bool)
    flags = rng.random(count) < 0.25
    if flags.all():
        flags[int(rng.integers(count))] = False
    if not flags.any() and rng.random() < 0.4:
        flags[int(rng.integers(count))] = True
    return flags


def _random_payload(
    inner_h: int, inner_w: int, open_dir: str, rng: np.random.Generator
) -> np.ndarray:
    area = inner_h * inner_w
    max_cells = max(1, min(area - 1 if area > 1 else 1, int(np.ceil(area * 0.6))))
    min_cells = 1 if area == 1 else 2 if rng.random() < 0.7 else 1
    min_cells = min(min_cells, max_cells)
    cell_count = int(rng.integers(min_cells, max_cells + 1))
    picks = rng.choice(area, size=cell_count, replace=False)

    patch = np.zeros((inner_h, inner_w), dtype=int)
    color_count = int(rng.integers(1, min(3, cell_count) + 1))
    colors = rng.choice(PAYLOAD_COLORS, size=color_count, replace=False)
    assignments = rng.choice(colors, size=cell_count, replace=True)
    for index, color in zip(picks, assignments):
        rr, cc = divmod(int(index), inner_w)
        patch[rr, cc] = int(color)

    axis = 0 if open_dir in {"up", "down"} else 1
    axis_len = inner_h if axis == 0 else inner_w
    coords = np.argwhere(patch != 0)
    if coords.shape[0] >= 2 and axis_len >= 2 and np.unique(coords[:, axis]).size == 1:
        for _ in range(20):
            rr = int(rng.integers(inner_h))
            cc = int(rng.integers(inner_w))
            if patch[rr, cc] != 0:
                continue
            if axis == 0 and rr == int(coords[0, 0]):
                continue
            if axis == 1 and cc == int(coords[0, 1]):
                continue
            patch[rr, cc] = int(rng.choice(colors))
            break

    return patch


def _draw_container(
    grid: np.ndarray,
    reserved: np.ndarray,
    r0: int,
    c0: int,
    height: int,
    width: int,
    open_dir: str,
    payload: np.ndarray | None,
) -> bool:
    rr0, cc0, rr1, cc1 = _reserve_box(r0, c0, height, width, open_dir)
    if rr1 >= grid.shape[0] or cc1 >= grid.shape[1]:
        return False
    if reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
        return False

    wall_mask = _container_template(height, width, open_dir)
    patch = grid[r0 : r0 + height, c0 : c0 + width]
    patch[wall_mask] = GEN_CONTAINER
    if payload is not None:
        interior = patch[1:-1, 1:-1]
        interior[payload != 0] = payload[payload != 0]

    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    count = int(rng.integers(1, 5))
    size_floor = 12 + 3 * count
    grid_h = int(rng.integers(size_floor, GRID_MAX + 1))
    grid_w = int(rng.integers(size_floor, GRID_MAX + 1))
    grid = np.full((grid_h, grid_w), GEN_BG, dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    directions = list(rng.permutation(DIRECTIONS)[:count])
    empty_flags = _empty_flags(count, rng)

    for idx in range(count):
        open_dir = directions[idx]
        is_empty = bool(empty_flags[idx])
        placed = False
        for _ in range(PLACE_RETRIES):
            height = int(rng.integers(CONTAINER_MIN, CONTAINER_MAX + 1))
            width = int(rng.integers(CONTAINER_MIN, CONTAINER_MAX + 1))
            ranges = _placement_ranges(grid_h, grid_w, height, width, open_dir)
            if ranges is None:
                continue
            (r_min, r_max), (c_min, c_max) = ranges
            r0 = int(rng.integers(r_min, r_max + 1))
            c0 = int(rng.integers(c_min, c_max + 1))
            payload = None
            if not is_empty:
                payload = _random_payload(height - 2, width - 2, open_dir, rng)
            if _draw_container(grid, reserved, r0, c0, height, width, open_dir, payload):
                placed = True
                break
        if not placed:
            return None

    containers = _find_containers(grid, GEN_BG)
    if len(containers) != count:
        return None

    transformed = main(grid)
    if np.array_equal(transformed, grid):
        return None
    if transformed.shape != grid.shape or max(transformed.shape) > GRID_MAX:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to create a valid container puzzle")
