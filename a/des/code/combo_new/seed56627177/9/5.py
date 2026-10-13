import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_RNG = np.random.default_rng()

GEN_RETRIES = 300
GRID_MIN = 15
GRID_MAX = 29
OBJECT_COUNT_CHOICES = np.array([2, 3, 4], dtype=int)
OBJECT_COUNT_WEIGHTS = np.array([3, 4, 2], dtype=float)
MAX_PROFILE_SPAN = 7
MAX_PROFILE_THICKNESS = 3
PLACEMENT_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for dr, dc in DIRS4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _analyze_component(coords: list[tuple[int, int]]) -> tuple[str, tuple[int, int, int, int]]:
    mask, bbox = _mask_from_coords(coords)
    is_horizontal = np.array_equal(mask, mask[::-1, :])
    is_vertical = np.array_equal(mask, mask[:, ::-1])
    assert is_horizontal ^ is_vertical, "each object must have exactly one symmetry axis"
    if is_horizontal:
        assert mask.shape[0] % 2 == 1, "horizontal-axis object must have a unique center row"
        return "horizontal", bbox
    assert mask.shape[1] % 2 == 1, "vertical-axis object must have a unique center column"
    return "vertical", bbox


def _projection_cells(
    bbox: tuple[int, int, int, int], axis: str, shape: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    if axis == "horizontal":
        span = c1 - c0 + 1
        center_r = (r0 + r1) // 2
        left_dist = c0
        right_dist = w - 1 - c1
        if left_dist <= right_dist:
            return [(center_r, c0 - k) for k in range(span)]
        return [(center_r, c1 + k) for k in range(span)]

    span = r1 - r0 + 1
    center_c = (c0 + c1) // 2
    top_dist = r0
    bottom_dist = h - 1 - r1
    if top_dist <= bottom_dist:
        return [(r0 - k, center_c) for k in range(span)]
    return [(r1 + k, center_c) for k in range(span)]


def main(grid: np.ndarray) -> np.ndarray:
    """Project each symmetric colored object toward its nearest boundary along its own symmetry axis.

    Input structure:
    - A 2D grid with one dominant background color.
    - Every foreground object is a 4-connected monochrome component.
    - Each object has exactly one reflection symmetry axis: horizontal or vertical.
    - Objects are separated by background space and leave enough room for a projection ray.

    Rule:
    - For each object, determine whether its unique symmetry axis is horizontal or vertical.
    - Let S be the object's span perpendicular to that axis: width for a horizontal axis, height for a vertical axis.
    - Take the center pixel on the bounding-box side that faces the nearest grid boundary along the axis direction, then paint a 1-pixel-wide ray of length S from that start cell toward that boundary in the object's color.
    - Keep every original object unchanged.

    Key insight:
    - The symmetry axis decides both the projection direction and which bounding-box dimension becomes the ray length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    out = g.copy()
    for comp in _components(g != bg):
        color = {int(g[r, c]) for r, c in comp}
        assert len(color) == 1, "each object must be monochrome"
        axis, bbox = _analyze_component(comp)
        for r, c in _projection_cells(bbox, axis, g.shape):
            assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1]
            out[r, c] = next(iter(color))
    return out


def _profile_mask_horizontal(width: int, max_depth: int) -> np.ndarray:
    for _ in range(200):
        depths = _RNG.integers(0, max_depth + 1, size=width)
        if int(depths.max()) == 0:
            continue
        if np.array_equal(depths, depths[::-1]):
            continue
        half = int(depths.max())
        mask = np.zeros((2 * half + 1, width), dtype=bool)
        center = half
        for c, depth in enumerate(depths):
            d = int(depth)
            mask[center - d : center + d + 1, c] = True
        if np.array_equal(mask, mask[:, ::-1]):
            continue
        return mask
    raise RuntimeError("failed to build a horizontal-axis object")


def _profile_mask_vertical(height: int, max_half_width: int) -> np.ndarray:
    for _ in range(200):
        half_widths = _RNG.integers(0, max_half_width + 1, size=height)
        if int(half_widths.max()) == 0:
            continue
        if np.array_equal(half_widths, half_widths[::-1]):
            continue
        half = int(half_widths.max())
        mask = np.zeros((height, 2 * half + 1), dtype=bool)
        center = half
        for r, hw in enumerate(half_widths):
            d = int(hw)
            mask[r, center - d : center + d + 1] = True
        if np.array_equal(mask, mask[::-1, :]):
            continue
        return mask
    raise RuntimeError("failed to build a vertical-axis object")


def _make_object_mask(axis: str, canvas_shape: tuple[int, int]) -> np.ndarray:
    h, w = canvas_shape
    if axis == "horizontal":
        max_width = min(MAX_PROFILE_SPAN, (w + 1) // 3)
        max_depth = min(MAX_PROFILE_THICKNESS, (h - 3) // 2)
        assert max_width >= 3 and max_depth >= 1
        width = int(_RNG.integers(3, max_width + 1))
        return _profile_mask_horizontal(width, max_depth)

    max_height = min(MAX_PROFILE_SPAN, (h + 1) // 3)
    max_half_width = min(MAX_PROFILE_THICKNESS, (w - 3) // 2)
    assert max_height >= 3 and max_half_width >= 1
    height = int(_RNG.integers(3, max_height + 1))
    return _profile_mask_vertical(height, max_half_width)


def _choose_axes(count: int) -> list[str]:
    n_horizontal = count // 2
    if count % 2 == 1:
        n_horizontal += int(_RNG.integers(0, 2))
    axes = ["horizontal"] * n_horizontal + ["vertical"] * (count - n_horizontal)
    _RNG.shuffle(axes)
    return axes


def _place_bbox(axis: str, mask_shape: tuple[int, int], canvas_shape: tuple[int, int]) -> tuple[int, int]:
    h, w = canvas_shape
    obj_h, obj_w = mask_shape
    if axis == "horizontal":
        if int(_RNG.integers(0, 2)) == 0:
            c0 = obj_w - 1
        else:
            c0 = w - 2 * obj_w + 1
        r0 = int(_RNG.integers(1, h - obj_h))
        return r0, c0

    if int(_RNG.integers(0, 2)) == 0:
        r0 = obj_h - 1
    else:
        r0 = h - 2 * obj_h + 1
    c0 = int(_RNG.integers(1, w - obj_w))
    return r0, c0


def _object_cells(mask: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    rows, cols = np.where(mask)
    return [(r0 + int(r), c0 + int(c)) for r, c in zip(rows, cols)]


def _blocked_mask(shape: tuple[int, int], cells: set[tuple[int, int]]) -> np.ndarray:
    h, w = shape
    blocked = np.zeros(shape, dtype=bool)
    for r, c in cells:
        blocked[r, c] = True
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True
    return blocked


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(GRID_MIN, GRID_MAX + 1))
    w = int(_RNG.integers(GRID_MIN, GRID_MAX + 1))
    bg = int(_RNG.integers(0, 10))
    count = int(_RNG.choice(OBJECT_COUNT_CHOICES, p=OBJECT_COUNT_WEIGHTS / OBJECT_COUNT_WEIGHTS.sum()))
    colors = [color for color in range(10) if color != bg]
    _RNG.shuffle(colors)
    axes = _choose_axes(count)

    grid = np.full((h, w), bg, dtype=int)
    reserved: set[tuple[int, int]] = set()
    blocked = np.zeros((h, w), dtype=bool)

    for color, axis in zip(colors, axes):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            try:
                mask = _make_object_mask(axis, (h, w))
            except AssertionError:
                return None
            obj_h, obj_w = mask.shape
            if axis == "horizontal" and w < 3 * obj_w - 1:
                continue
            if axis == "vertical" and h < 3 * obj_h - 1:
                continue

            r0, c0 = _place_bbox(axis, mask.shape, (h, w))
            coords = _object_cells(mask, r0, c0)
            bbox = (r0, c0, r0 + obj_h - 1, c0 + obj_w - 1)
            ray = _projection_cells(bbox, axis, (h, w))
            cells = set(coords) | set(ray)
            if any(blocked[r, c] for r, c in cells):
                continue

            for r, c in coords:
                grid[r, c] = color
            reserved |= cells
            blocked = _blocked_mask((h, w), reserved)
            placed = True
            break
        if not placed:
            return None

    comps = _components(grid != bg)
    if len(comps) != count:
        return None
    axes_seen = []
    for comp in comps:
        comp_colors = {int(grid[r, c]) for r, c in comp}
        if len(comp_colors) != 1:
            return None
        axis, bbox = _analyze_component(comp)
        axes_seen.append(axis)
        ray = _projection_cells(bbox, axis, grid.shape)
        end_r, end_c = ray[-1]
        if end_r not in (0, h - 1) and end_c not in (0, w - 1):
            return None

    if "horizontal" not in axes_seen or "vertical" not in axes_seen:
        return None

    main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
