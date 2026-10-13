from __future__ import annotations

from collections import Counter, deque

import numpy as np


BACKGROUND = 0
TRANSFORM_NAMES = (
    "mirror_horizontal",
    "mirror_vertical",
    "rotate_cw",
    "rotate_ccw",
    "rotate_180",
)
GEN_RETRIES = 400
SHAPE_RETRIES = 300
PLACE_RETRIES = 300
RNG = np.random.default_rng()
_TRANSFORM_DECK: list[str] = []


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _connected_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    height, width = map(int, grid.shape)
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            coord_array = np.asarray(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coord_array,
                    "bbox": _bbox_from_coords(coord_array),
                }
            )

    return components


def _component_mask(component: dict[str, object]) -> np.ndarray:
    coords = component["coords"]
    if not isinstance(coords, np.ndarray):
        raise TypeError("component coords must be an ndarray")
    r0, c0, r1, c1 = component["bbox"]
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    mask = np.zeros((height, width), dtype=bool)
    shifted = coords - np.array([[r0, c0]], dtype=int)
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask


def _apply_transform(array: np.ndarray, transform: str) -> np.ndarray:
    if transform == "mirror_horizontal":
        return np.flipud(array)
    if transform == "mirror_vertical":
        return np.fliplr(array)
    if transform == "rotate_cw":
        return np.rot90(array, -1)
    if transform == "rotate_ccw":
        return np.rot90(array, 1)
    if transform == "rotate_180":
        return np.rot90(array, 2)
    raise ValueError(f"unknown transform: {transform}")


def _reading_key(component: dict[str, object]) -> tuple[int, int, int, int]:
    bbox = component["bbox"]
    return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])


def _infer_demo_transform(demo_pair: list[dict[str, object]]) -> str:
    assert len(demo_pair) == 2, "demo pair must contain exactly two sprites"
    source, target = sorted(demo_pair, key=_reading_key)
    source_mask = _component_mask(source)
    target_mask = _component_mask(target)
    matches = [
        name
        for name in TRANSFORM_NAMES
        if np.array_equal(_apply_transform(source_mask, name), target_mask)
    ]
    assert len(matches) == 1, "same-color demo pair must encode a unique transform"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the rigid motion shown by the same-color demo pair and apply it in place to the third sprite.

    Input structure:
    - Black is the background color.
    - Exactly three disconnected monochrome sprites appear on the canvas.
    - Two sprites share one color and form the demo pair; the third sprite has a
      different color and a square bounding box.

    Rule:
    - Find the two same-color demo sprites and order them by top row, then left column.
    - Among horizontal mirror, vertical mirror, clockwise 90-degree rotation,
      counterclockwise 90-degree rotation, and 180-degree rotation, identify the
      unique transform that maps the earlier demo sprite to the later one.
    - Clear the third sprite's square bounding box and redraw that sprite after
      applying the same transform around its own bounding-box center.
    - Leave the demo pair, background, and query color unchanged otherwise.

    Key insight:
    - The repeated-color pair is an in-grid demonstration of pose change; its
      spatial offset is irrelevant, and only the orientation transform transfers
      to the differently colored query sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    bg = _most_common_color(g)
    components = _connected_components(g, bg)
    assert len(components) == 3, "expected exactly three foreground sprites"

    by_color: dict[int, list[dict[str, object]]] = {}
    for component in components:
        by_color.setdefault(int(component["color"]), []).append(component)

    demo_groups = [group for group in by_color.values() if len(group) == 2]
    query_groups = [group for group in by_color.values() if len(group) == 1]
    assert len(demo_groups) == 1 and len(query_groups) == 1, (
        "expected exactly one repeated-color demo pair and one differently colored query sprite"
    )

    transform = _infer_demo_transform(demo_groups[0])
    query = query_groups[0][0]
    r0, c0, r1, c1 = query["bbox"]
    assert int(r1 - r0) == int(c1 - c0), "query sprite must have a square bounding box"

    patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    moved = _apply_transform(patch, transform)
    assert moved.shape == patch.shape, "query transform must stay inside the square bbox"

    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = bg
    out[r0 : r1 + 1, c0 : c1 + 1] = moved
    return out


def _choose_transform() -> str:
    global _TRANSFORM_DECK
    if not _TRANSFORM_DECK:
        _TRANSFORM_DECK = list(TRANSFORM_NAMES)
        RNG.shuffle(_TRANSFORM_DECK)
    return _TRANSFORM_DECK.pop()


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].copy()


def _mask_from_cells(shape: tuple[int, int], cells: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _grow_connected_mask(height: int, width: int, min_cells: int, max_cells: int) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        target = int(RNG.integers(min_cells, max_cells + 1))
        start = (int(RNG.integers(height)), int(RNG.integers(width)))
        cells = {start}

        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = frontier[int(RNG.integers(len(frontier)))]
            cells.add((nr, nc))

        if len(cells) != target:
            continue

        mask = _crop_mask(_mask_from_cells((height, width), cells))
        if mask.size > 0:
            return mask

    raise RuntimeError("failed to build a connected mask")


def _generate_demo_mask(transform: str) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        height = int(RNG.integers(2, 6))
        width = int(RNG.integers(2, 6))
        max_cells = min(height * width - 1, 8)
        if max_cells < 3:
            continue

        mask = _grow_connected_mask(height, width, 3, max_cells)
        moved = _apply_transform(mask, transform)
        if np.array_equal(mask, moved):
            continue

        matches = [
            name
            for name in TRANSFORM_NAMES
            if np.array_equal(_apply_transform(mask, name), moved)
        ]
        if matches == [transform]:
            return mask

    raise RuntimeError("failed to generate an asymmetric demo sprite")


def _random_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    er, ec = end
    path = [(r, c)]

    while (r, c) != (er, ec):
        steps: list[tuple[int, int]] = []
        if r < er:
            steps.append((r + 1, c))
        elif r > er:
            steps.append((r - 1, c))
        if c < ec:
            steps.append((r, c + 1))
        elif c > ec:
            steps.append((r, c - 1))
        r, c = steps[int(RNG.integers(len(steps)))]
        path.append((r, c))

    return path


def _generate_query_mask(size: int, transform: str) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        anchors = [
            (0, int(RNG.integers(size))),
            (size - 1, int(RNG.integers(size))),
            (int(RNG.integers(size)), 0),
            (int(RNG.integers(size)), size - 1),
        ]
        order = list(range(len(anchors)))
        RNG.shuffle(order)

        cells = {anchors[order[0]]}
        for idx in order[1:]:
            start = tuple(list(cells)[int(RNG.integers(len(cells)))])
            for cell in _random_path(start, anchors[idx]):
                cells.add(cell)

        target = int(RNG.integers(max(len(cells), size + 1), size * size))
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(RNG.integers(len(frontier)))])

        mask = _mask_from_cells((size, size), cells)
        if not (
            np.any(mask[0, :])
            and np.any(mask[-1, :])
            and np.any(mask[:, 0])
            and np.any(mask[:, -1])
        ):
            continue
        if np.array_equal(mask, _apply_transform(mask, transform)):
            continue
        return mask

    raise RuntimeError("failed to generate a query sprite")


def _colorize(mask: np.ndarray, color: int, bg: int = BACKGROUND) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = int(color)
    return patch


def _bbox_is_clear(
    candidate: tuple[int, int, int, int], existing: list[tuple[int, int, int, int]], gap: int = 1
) -> bool:
    r0, c0, r1, c1 = candidate
    for er0, ec0, er1, ec1 in existing:
        separated = (
            r1 + gap < er0
            or er1 + gap < r0
            or c1 + gap < ec0
            or ec1 + gap < c0
        )
        if not separated:
            return False
    return True


def _place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    existing: list[tuple[int, int, int, int]],
    *,
    require_after: tuple[int, int] | None = None,
) -> tuple[int, int, int, int]:
    height, width = map(int, grid.shape)
    ph, pw = map(int, patch.shape)

    for _ in range(PLACE_RETRIES):
        r0 = int(RNG.integers(0, height - ph + 1))
        c0 = int(RNG.integers(0, width - pw + 1))
        if require_after is not None and (r0, c0) <= require_after:
            continue

        bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        if not _bbox_is_clear(bbox, existing, gap=1):
            continue

        window = grid[r0 : r0 + ph, c0 : c0 + pw]
        window[patch != BACKGROUND] = patch[patch != BACKGROUND]
        existing.append(bbox)
        return bbox

    raise RuntimeError("failed to place patch")


def _attempt_generate() -> np.ndarray | None:
    transform = _choose_transform()
    demo_mask = _generate_demo_mask(transform)
    query_size = int(RNG.integers(3, 7))
    query_mask = _generate_query_mask(query_size, transform)

    demo_src_color, query_color = [int(v) for v in RNG.choice(np.arange(1, 10), size=2, replace=False)]
    demo_src = _colorize(demo_mask, demo_src_color)
    demo_dst = _colorize(_apply_transform(demo_mask, transform), demo_src_color)
    query = _colorize(query_mask, query_color)

    placed = False
    for _ in range(PLACE_RETRIES):
        grid_height = int(RNG.integers(14, 31))
        grid_width = int(RNG.integers(14, 31))
        grid = np.full((grid_height, grid_width), BACKGROUND, dtype=int)
        bboxes: list[tuple[int, int, int, int]] = []
        try:
            demo_src_bbox = _place_patch(grid, demo_src, bboxes)
            _place_patch(grid, demo_dst, bboxes, require_after=(demo_src_bbox[0], demo_src_bbox[1]))
            _place_patch(grid, query, bboxes)
        except RuntimeError:
            continue
        placed = True
        break

    if not placed:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None

    components = _connected_components(grid, BACKGROUND)
    if len(components) != 3:
        return None

    by_color: dict[int, int] = {}
    for component in components:
        color = int(component["color"])
        by_color[color] = by_color.get(color, 0) + 1
    if sorted(by_color.values()) != [1, 2]:
        return None

    query_component = next(component for component in components if by_color[int(component["color"])] == 1)
    qr0, qc0, qr1, qc1 = query_component["bbox"]
    if int(qr1 - qr0) != int(qc1 - qc0):
        return None

    expected_transform = _infer_demo_transform(
        [component for component in components if by_color[int(component["color"])] == 2]
    )
    if expected_transform != transform:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
