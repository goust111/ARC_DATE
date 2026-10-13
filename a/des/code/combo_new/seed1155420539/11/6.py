import random

import numpy as np


BG = 0
AZURE = 8
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRECTIONS = ("up", "down", "left", "right")
BODY_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
GEN_RETRIES = 400


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    components: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _erode(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return mask.copy()
    padded = np.pad(mask, 1, constant_values=False)
    eroded = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return _crop_mask(eroded)


def _shrink_sequence(mask: np.ndarray) -> list[np.ndarray]:
    cur = _crop_mask(mask)
    shrunk: list[np.ndarray] = []
    while True:
        cur = _erode(cur)
        if cur.size == 0 or not cur.any():
            return shrunk
        shrunk.append(cur)


def _direction_from_marker(mask: np.ndarray, marker: tuple[int, int]) -> str:
    r, c = marker
    h, w = mask.shape
    candidates: list[str] = []
    if r == 0 and h > 1 and mask[min(1, h - 1), c]:
        candidates.append("up")
    if r == h - 1 and h > 1 and mask[max(0, h - 2), c]:
        candidates.append("down")
    if c == 0 and w > 1 and mask[r, min(1, w - 1)]:
        candidates.append("left")
    if c == w - 1 and w > 1 and mask[r, max(0, w - 2)]:
        candidates.append("right")
    assert len(candidates) == 1, "azure marker must identify exactly one outward side"
    return candidates[0]


def _component_spec(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int], str, int]:
    arr = np.array(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    body_colors: set[int] = set()
    markers: list[tuple[int, int]] = []
    for r, c in coords:
        value = int(grid[r, c])
        local[r - r0, c - c0] = True
        if value == AZURE:
            markers.append((r - r0, c - c0))
        else:
            body_colors.add(value)
    assert len(markers) == 1, "each object must contain exactly one azure marker"
    assert len(body_colors) == 1, "each object must have a single body color"
    direction = _direction_from_marker(local, markers[0])
    return local, (int(r0), int(c0)), direction, next(iter(body_colors))


def _copy_layouts(mask: np.ndarray, origin: tuple[int, int], direction: str) -> list[tuple[np.ndarray, int, int]]:
    base_r, base_c = origin
    base_h, base_w = mask.shape
    layouts: list[tuple[np.ndarray, int, int]] = []
    front = {
        "up": base_r,
        "down": base_r + base_h - 1,
        "left": base_c,
        "right": base_c + base_w - 1,
    }[direction]
    for inner in _shrink_sequence(mask):
        h, w = inner.shape
        if direction == "up":
            left = base_c + (base_w - w) // 2
            bottom = front - 2
            top = bottom - h + 1
            front = top
        elif direction == "down":
            left = base_c + (base_w - w) // 2
            top = front + 2
            front = top + h - 1
        elif direction == "left":
            top = base_r + (base_h - h) // 2
            right = front - 2
            left = right - w + 1
            front = left
        else:
            top = base_r + (base_h - h) // 2
            left = front + 2
            front = left + w - 1
        layouts.append((inner, int(top), int(left)))
    return layouts


def _paint_mask(out: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    height, width = out.shape
    for r, c in np.argwhere(mask):
        rr = top + int(r)
        cc = left + int(c)
        if 0 <= rr < height and 0 <= cc < width:
            out[rr, cc] = int(color)


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()
    for coords in _find_components(g != BG):
        mask, origin, direction, body_color = _component_spec(g, coords)
        for inner, top, left in _copy_layouts(mask, origin, direction):
            _paint_mask(out, inner, top, left, body_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp progressively eroded copies of each marked object in the marker's outward direction.

    Input structure:
    - The background is black.
    - The foreground consists of several 4-connected solid convex objects, each with one body color.
    - In every object, exactly one body pixel is replaced by an azure marker located at the midpoint of one bbox edge.

    Rule:
    - For each object, read the azure marker's edge to determine the outward direction: top, bottom, left, or right.
    - Repeatedly erode the full object mask by one 4-neighborhood layer to obtain progressively smaller inner copies.
    - Place those copies in front of the original object, keeping a one-pixel gap between consecutive copies and centering every copy along the perpendicular axis; paint them in the object's body color while keeping the original object unchanged.

    Key insight:
    - The azure pixel does not change the shape identity; it marks which side is "forward", and the rest of the task is an iterative inner-shell erosion plus directional stamping layout.
    """
    return _apply_rule(grid)


def _rectangle_mask() -> np.ndarray:
    height = random.choice((3, 5, 7, 9))
    width = random.choice((3, 5, 7, 9))
    return np.ones((height, width), dtype=bool)


def _diamond_mask() -> np.ndarray:
    radius = random.randint(1, 4)
    rows, cols = np.indices((2 * radius + 1, 2 * radius + 1))
    center = radius
    return (np.abs(rows - center) + np.abs(cols - center) <= radius)


def _sample_shape() -> np.ndarray:
    kind = random.choice(("rectangle", "diamond"))
    return _rectangle_mask() if kind == "rectangle" else _diamond_mask()


def _marker_position(mask: np.ndarray, direction: str) -> tuple[int, int]:
    h, w = mask.shape
    if direction == "up":
        return 0, w // 2
    if direction == "down":
        return h - 1, w // 2
    if direction == "left":
        return h // 2, 0
    return h // 2, w - 1


def _render_object(mask: np.ndarray, direction: str, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    mr, mc = _marker_position(mask, direction)
    patch[mr, mc] = AZURE
    return patch


def _footprint_cells(mask: np.ndarray, direction: str) -> set[tuple[int, int]]:
    cells = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    for inner, top, left in _copy_layouts(mask, (0, 0), direction):
        for r, c in np.argwhere(inner):
            cells.add((top + int(r), left + int(c)))
    return cells


def _padding_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    height, width = shape
    padded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    padded.add((rr, cc))
    return padded


def _placeable_rows_cols(
    footprint: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[list[int], list[int]]:
    arr = np.array(list(footprint), dtype=int)
    min_r, min_c = arr.min(axis=0)
    max_r, max_c = arr.max(axis=0)
    height, width = shape
    rows = list(range(-int(min_r), height - int(max_r)))
    cols = list(range(-int(min_c), width - int(max_c)))
    return rows, cols


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(18, 30)
    width = random.randint(18, 30)
    count = random.randint(1, 4)
    colors = random.sample(BODY_COLORS, count)
    grid = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    for color in colors:
        placed = False
        for _ in range(80):
            mask = _sample_shape()
            direction = random.choice(DIRECTIONS)
            footprint = _footprint_cells(mask, direction)
            rows, cols = _placeable_rows_cols(footprint, (height, width))
            if not rows or not cols:
                continue
            top = random.choice(rows)
            left = random.choice(cols)
            shifted = {(top + r, left + c) for r, c in footprint}
            if any(blocked[r, c] for r, c in shifted):
                continue
            patch = _render_object(mask, direction, color)
            ph, pw = patch.shape
            grid[top : top + ph, left : left + pw][patch != BG] = patch[patch != BG]
            for r, c in _padding_cells(shifted, grid.shape):
                blocked[r, c] = True
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
