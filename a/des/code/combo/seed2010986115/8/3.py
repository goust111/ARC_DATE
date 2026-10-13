from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
AZURE = 8
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Four 2x2 L-shaped arrow orientations. The prompt text for "down" is inconsistent;
# this mapping keeps the four orientations distinct and matches the left/right examples.
ARROW_PATTERNS: dict[str, frozenset[tuple[int, int]]] = {
    "right": frozenset({(1, 1), (1, 0), (0, 1)}),  # elbow at bottom-right
    "left": frozenset({(1, 0), (1, 1), (0, 0)}),   # elbow at bottom-left
    "up": frozenset({(0, 1), (1, 1), (0, 0)}),     # elbow at top-right
    "down": frozenset({(0, 0), (1, 0), (0, 1)}),   # elbow at top-left
}
HORIZONTAL_FLIP_DIRS = {"left", "right"}

QUADRANT_SIZE_RANGE = (8, 13)
OBJECT_AREA_MIN = 4
OBJECT_AREA_MAX = 18
OBJECT_BBOX_MAX = 6
MAX_GENERATION_ATTEMPTS = 300
OBJECT_COLORS = [1, 2, 3, 5, 6, 7, 9]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rs = coords[:, 0]
    cs = coords[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _normalize_coords(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return frozenset((int(r - r0), int(c - c0)) for r, c in coords)


def _find_divider(g: np.ndarray) -> tuple[int, int]:
    h, w = g.shape
    rows = [r for r in range(h) if np.all(g[r, :] == AZURE)]
    cols = [c for c in range(w) if np.all(g[:, c] == AZURE)]
    assert len(rows) == 1 and len(cols) == 1, "expected one azure row and one azure column"
    mid_r, mid_c = int(rows[0]), int(cols[0])
    assert mid_r == h - mid_r - 1, "horizontal divider must split equal-height quadrants"
    assert mid_c == w - mid_c - 1, "vertical divider must split equal-width quadrants"
    return mid_r, mid_c


def _flip_axis_for_arrow(coords: np.ndarray) -> str:
    norm = _normalize_coords(coords)
    matches = [name for name, pattern in ARROW_PATTERNS.items() if pattern == norm]
    assert len(matches) == 1, "yellow component must be one valid arrow"
    direction = matches[0]
    return "horizontal" if direction in HORIZONTAL_FLIP_DIRS else "vertical"


def _extract_transformed_object(quadrant: np.ndarray) -> dict[str, np.ndarray | int]:
    fg = (quadrant != BLACK) & (quadrant != AZURE)
    comps = _components(fg)
    assert len(comps) == 2, "each quadrant must contain exactly one object and one arrow"

    arrow: np.ndarray | None = None
    obj: np.ndarray | None = None
    for comp in comps:
        colors = {int(quadrant[int(r), int(c)]) for r, c in comp}
        if len(comp) == 3 and colors == {YELLOW} and _normalize_coords(comp) in set(ARROW_PATTERNS.values()):
            assert arrow is None, "arrow must be unique"
            arrow = comp
        else:
            assert obj is None, "object must be unique"
            obj = comp

    assert arrow is not None and obj is not None

    obj_colors = {int(quadrant[int(r), int(c)]) for r, c in obj}
    assert len(obj_colors) == 1, "object must be monochrome"
    color = next(iter(obj_colors))

    r0, c0, r1, c1 = _bbox(obj)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
    for r, c in obj:
        patch[int(r - r0), int(c - c0)] = color

    if _flip_axis_for_arrow(arrow) == "horizontal":
        patch = np.fliplr(patch)
    else:
        patch = np.flipud(patch)

    return {"area": int(len(obj)), "patch": patch}


def _render_centered(objects: list[dict[str, np.ndarray | int]], shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    canvas_h, canvas_w = map(int, shape)
    for item in objects:
        patch = np.asarray(item["patch"], dtype=int)
        ph, pw = map(int, patch.shape)
        r0 = (canvas_h - ph) // 2
        c0 = (canvas_w - pw) // 2
        mask = patch != BLACK
        sub = out[r0 : r0 + ph, c0 : c0 + pw]
        sub[mask] = patch[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each quadrant object according to its local yellow arrow, then overlay the
    mirrored objects on a centered canvas in descending area order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    mid_r, mid_c = _find_divider(g)
    quadrant_h = int(mid_r)
    quadrant_w = int(mid_c)
    quadrants = [
        g[:mid_r, :mid_c],
        g[:mid_r, mid_c + 1 :],
        g[mid_r + 1 :, :mid_c],
        g[mid_r + 1 :, mid_c + 1 :],
    ]
    assert all(q.shape == (quadrant_h, quadrant_w) for q in quadrants)

    objects = [_extract_transformed_object(quadrant) for quadrant in quadrants]
    objects.sort(key=lambda item: int(item["area"]), reverse=True)
    return _render_centered(objects, (quadrant_h, quadrant_w))


def _dilate4(mask: np.ndarray) -> np.ndarray:
    dilated = mask.copy()
    dilated[1:, :] |= mask[:-1, :]
    dilated[:-1, :] |= mask[1:, :]
    dilated[:, 1:] |= mask[:, :-1]
    dilated[:, :-1] |= mask[:, 1:]
    return dilated


def _grow_mask(height: int, width: int, area: int) -> np.ndarray:
    assert 1 <= area <= height * width
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < area:
        frontier = {
            (r + dr, c + dc)
            for r, c in cells
            for dr, dc in DIR4
            if 0 <= r + dr < height and 0 <= c + dc < width and (r + dr, c + dc) not in cells
        }
        pick = random.choice(tuple(frontier))
        cells.add((int(pick[0]), int(pick[1])))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_object_mask(qh: int, qw: int, area: int, flip_axis: str) -> np.ndarray | None:
    max_h = min(qh - 2, OBJECT_BBOX_MAX)
    max_w = min(qw - 2, OBJECT_BBOX_MAX)
    dims = [(h, w) for h in range(2, max_h + 1) for w in range(2, max_w + 1) if area <= h * w]
    if not dims:
        return None

    irregular = [dim for dim in dims if dim[0] * dim[1] > area]
    pool = irregular if irregular else dims

    for _ in range(200):
        height, width = random.choice(pool)
        mask = _crop_mask(_grow_mask(height, width, area))
        if mask.shape[0] > qh or mask.shape[1] > qw:
            continue
        if flip_axis == "horizontal" and np.array_equal(mask, np.fliplr(mask)):
            continue
        if flip_axis == "vertical" and np.array_equal(mask, np.flipud(mask)):
            continue
        return mask
    return None


def _valid_arrow_origins(occupied: np.ndarray, direction: str) -> list[tuple[int, int]]:
    qh, qw = occupied.shape
    blocked = _dilate4(occupied)
    origins: list[tuple[int, int]] = []
    for r0 in range(qh - 1):
        for c0 in range(qw - 1):
            cells = [(r0 + dr, c0 + dc) for dr, dc in ARROW_PATTERNS[direction]]
            if any(blocked[r, c] for r, c in cells):
                continue
            origins.append((int(r0), int(c0)))
    return origins


def _build_quadrant(
    qh: int,
    qw: int,
    color: int,
    area: int,
    direction: str,
) -> tuple[np.ndarray, dict[str, np.ndarray | int]] | None:
    flip_axis = "horizontal" if direction in HORIZONTAL_FLIP_DIRS else "vertical"

    for _ in range(200):
        mask = _random_object_mask(qh, qw, area, flip_axis)
        if mask is None:
            return None

        oh, ow = map(int, mask.shape)
        positions = [(r0, c0) for r0 in range(qh - oh + 1) for c0 in range(qw - ow + 1)]
        random.shuffle(positions)

        for r0, c0 in positions:
            occupied = np.zeros((qh, qw), dtype=bool)
            occupied[r0 : r0 + oh, c0 : c0 + ow] |= mask
            arrow_origins = _valid_arrow_origins(occupied, direction)
            if not arrow_origins:
                continue

            ar0, ac0 = random.choice(arrow_origins)
            quadrant = np.full((qh, qw), BLACK, dtype=int)
            sub = quadrant[r0 : r0 + oh, c0 : c0 + ow]
            sub[mask] = color
            for dr, dc in ARROW_PATTERNS[direction]:
                quadrant[ar0 + dr, ac0 + dc] = YELLOW

            patch = np.full((oh, ow), BLACK, dtype=int)
            patch[mask] = color
            transformed = np.fliplr(patch) if flip_axis == "horizontal" else np.flipud(patch)
            return quadrant, {"area": int(area), "patch": transformed}
    return None


def _assemble_grid(quadrants: list[np.ndarray]) -> np.ndarray:
    qh, qw = map(int, quadrants[0].shape)
    grid = np.full((2 * qh + 1, 2 * qw + 1), BLACK, dtype=int)
    grid[qh, :] = AZURE
    grid[:, qw] = AZURE
    grid[:qh, :qw] = quadrants[0]
    grid[:qh, qw + 1 :] = quadrants[1]
    grid[qh + 1 :, :qw] = quadrants[2]
    grid[qh + 1 :, qw + 1 :] = quadrants[3]
    return grid


def _gen_attempt() -> np.ndarray | None:
    qh = random.randint(*QUADRANT_SIZE_RANGE)
    qw = random.randint(*QUADRANT_SIZE_RANGE)
    colors = random.sample(OBJECT_COLORS, 4)
    areas = random.sample(range(OBJECT_AREA_MIN, OBJECT_AREA_MAX + 1), 4)
    directions = random.sample(list(ARROW_PATTERNS.keys()), 4)

    quadrants: list[np.ndarray] = []
    objects: list[dict[str, np.ndarray | int]] = []
    for color, area, direction in zip(colors, areas, directions, strict=True):
        built = _build_quadrant(qh, qw, color, area, direction)
        if built is None:
            return None
        quadrant, obj = built
        quadrants.append(quadrant)
        objects.append(obj)

    expected = _render_centered(sorted(objects, key=lambda item: int(item["area"]), reverse=True), (qh, qw))
    reversed_order = _render_centered(sorted(objects, key=lambda item: int(item["area"])), (qh, qw))
    if np.array_equal(expected, reversed_order):
        return None

    grid = _assemble_grid(quadrants)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
