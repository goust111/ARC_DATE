import numpy as np
from collections import deque


ORANGE = 7
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

LEGEND_HEIGHT = 3
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
RNG = np.random.default_rng()

GEN_ATTEMPTS = 300
SHAPE_ATTEMPTS = 200
PLACEMENT_MARGIN = 1
PLACEMENT_GAP = 1
BODY_HEIGHT_RANGE = (11, 19)
BODY_WIDTH_RANGE = (20, 29)
OBJECT_COUNT_RANGE = (3, 7)

LEGEND_HORIZONTAL = np.array(
    [
        [1, 1, 0],
        [0, 1, 0],
        [1, 1, 0],
    ],
    dtype=bool,
)
LEGEND_VERTICAL = LEGEND_HORIZONTAL.T.copy()
LEGEND_NONE = np.array(
    [
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 1],
    ],
    dtype=bool,
)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    assert rows.size > 0
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _tight_mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[rows - r0, cols - c0] = True
    return mask, (r0, c0, r1, c1)


def _paste_mask(canvas: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    assert 0 <= top <= canvas.shape[0] - h
    assert 0 <= left <= canvas.shape[1] - w
    region = canvas[top : top + h, left : left + w]
    assert np.all(region[mask] == ORANGE)
    region[mask] = color


def _symmetry_type(mask: np.ndarray) -> str:
    horizontal = np.array_equal(mask, mask[::-1, :])
    vertical = np.array_equal(mask, mask[:, ::-1])
    if horizontal and not vertical:
        return "horizontal"
    if vertical and not horizontal:
        return "vertical"
    if not horizontal and not vertical:
        return "none"
    raise AssertionError("object symmetry is ambiguous")


def _rotate_cw_about_bbox_center(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = mask.shape
    assert h % 2 == w % 2, "90-degree rotation around the bbox center must stay on-grid"
    return np.rot90(mask, -1), (h - w) // 2, (w - h) // 2


def _transform_object(mask: np.ndarray, sym_type: str) -> tuple[np.ndarray, int, int, int]:
    if sym_type == "horizontal":
        rotated, dr, dc = _rotate_cw_about_bbox_center(mask)
        return rotated, dr, dc, RED
    if sym_type == "vertical":
        return np.fliplr(mask), 0, 0, BLUE
    if sym_type == "none":
        return mask.copy(), 0, 0, GREEN
    raise AssertionError(f"unknown symmetry type: {sym_type}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect each grey foreground object's symmetry type, apply the legend-selected transformation,
    and drop the top legend area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] > LEGEND_HEIGHT

    bg = _most_common_color(g)
    assert bg == ORANGE

    body = g[LEGEND_HEIGHT:, :]
    assert np.all((body == ORANGE) | (body == GREY))

    out = np.full_like(body, ORANGE)
    comps = _components(body == GREY)
    assert comps

    for coords in comps:
        mask, (r0, c0, _, _) = _tight_mask_from_coords(coords)
        sym_type = _symmetry_type(mask)
        transformed, dr, dc, color = _transform_object(mask, sym_type)
        _paste_mask(out, transformed, r0 + dr, c0 + dc, color)

    return out


def _mask_from_cells(cells: set[tuple[int, int]], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _grow_connected(height: int, width: int, target: int, start: tuple[int, int]) -> set[tuple[int, int]]:
    cells = {start}
    while len(cells) < target:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        frontier_list = list(frontier)
        cells.add(frontier_list[int(RNG.integers(len(frontier_list)))])
    return cells


def _generate_horizontal_only_mask() -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        height = int(RNG.integers(3, 7))
        width = int(RNG.integers(3, 7))
        if height % 2 != width % 2:
            continue
        half_height = (height + 1) // 2
        start = (half_height - 1, int(RNG.integers(width)))
        min_cells = max(2, min(width, half_height + 1))
        target = int(RNG.integers(min_cells, half_height * width + 1))
        base = _grow_connected(half_height, width, target, start)
        cells: set[tuple[int, int]] = set()
        for r, c in base:
            cells.add((r, c))
            cells.add((height - 1 - r, c))
        mask = _crop_mask(_mask_from_cells(cells, height, width))
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if mask.shape[0] % 2 != mask.shape[1] % 2:
            continue
        if mask.sum() < 4 or not _is_connected(mask):
            continue
        horizontal = np.array_equal(mask, mask[::-1, :])
        vertical = np.array_equal(mask, mask[:, ::-1])
        if horizontal and not vertical:
            return mask
    raise RuntimeError("failed to generate a horizontal-only mask")


def _generate_vertical_only_mask() -> np.ndarray:
    return _generate_horizontal_only_mask().T.copy()


def _generate_asymmetric_mask() -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        height = int(RNG.integers(3, 7))
        width = int(RNG.integers(3, 7))
        start = (int(RNG.integers(height)), int(RNG.integers(width)))
        min_cells = max(4, (height * width) // 4)
        max_cells = max(min_cells, (2 * height * width) // 3)
        target = int(RNG.integers(min_cells, max_cells + 1))
        mask = _crop_mask(_mask_from_cells(_grow_connected(height, width, target, start), height, width))
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if mask.sum() < 4 or not _is_connected(mask):
            continue
        horizontal = np.array_equal(mask, mask[::-1, :])
        vertical = np.array_equal(mask, mask[:, ::-1])
        if not horizontal and not vertical:
            return mask
    raise RuntimeError("failed to generate an asymmetric mask")


def _footprint(mask: np.ndarray, sym_type: str) -> dict[str, int]:
    h, w = mask.shape
    if sym_type == "horizontal":
        _, dr, dc = _rotate_cw_about_bbox_center(mask)
        out_h, out_w = w, h
    else:
        dr = dc = 0
        out_h, out_w = h, w

    min_r = min(0, dr)
    min_c = min(0, dc)
    max_r = max(h, dr + out_h)
    max_c = max(w, dc + out_w)
    return {
        "union_h": max_r - min_r,
        "union_w": max_c - min_c,
        "orig_r": -min_r,
        "orig_c": -min_c,
    }


def _balanced_types(count: int) -> list[str]:
    types = ["horizontal", "vertical", "none"]
    counts = {"horizontal": 1, "vertical": 1, "none": 1}
    while len(types) < count:
        min_count = min(counts.values())
        choices = [name for name, value in counts.items() if value == min_count]
        choice = choices[int(RNG.integers(len(choices)))]
        types.append(choice)
        counts[choice] += 1
    order = np.asarray(types, dtype=object)
    RNG.shuffle(order)
    return [str(item) for item in order]


def _place_objects(objects: list[dict], body_h: int, body_w: int) -> list[dict] | None:
    occupied = np.zeros((body_h, body_w), dtype=bool)
    placed: list[dict] = []
    order = sorted(objects, key=lambda item: item["fp"]["union_h"] * item["fp"]["union_w"], reverse=True)

    for obj in order:
        fp = obj["fp"]
        candidates: list[tuple[int, int]] = []
        max_row = body_h - fp["union_h"] - PLACEMENT_MARGIN
        max_col = body_w - fp["union_w"] - PLACEMENT_MARGIN
        if max_row < PLACEMENT_MARGIN or max_col < PLACEMENT_MARGIN:
            return None
        for union_r in range(PLACEMENT_MARGIN, max_row + 1):
            for union_c in range(PLACEMENT_MARGIN, max_col + 1):
                r0 = max(0, union_r - PLACEMENT_GAP)
                r1 = min(body_h, union_r + fp["union_h"] + PLACEMENT_GAP)
                c0 = max(0, union_c - PLACEMENT_GAP)
                c1 = min(body_w, union_c + fp["union_w"] + PLACEMENT_GAP)
                if not occupied[r0:r1, c0:c1].any():
                    candidates.append((union_r, union_c))
        if not candidates:
            return None
        union_r, union_c = candidates[int(RNG.integers(len(candidates)))]
        occupied[union_r : union_r + fp["union_h"], union_c : union_c + fp["union_w"]] = True
        placed.append(
            {
                "mask": obj["mask"],
                "sym_type": obj["sym_type"],
                "row": union_r + fp["orig_r"],
                "col": union_c + fp["orig_c"],
            }
        )

    return placed


def _draw_legend(grid: np.ndarray) -> None:
    width = grid.shape[1]
    legend_width = 16
    start = int(RNG.integers(0, width - legend_width + 1))

    grid[0, start] = RED
    grid[1, start + 6] = BLUE
    grid[2, start + 12] = GREEN

    for mask, left in (
        (LEGEND_HORIZONTAL, start + 1),
        (LEGEND_VERTICAL, start + 7),
        (LEGEND_NONE, start + 13),
    ):
        region = grid[:LEGEND_HEIGHT, left : left + 3]
        region[mask] = GREY


def _gen_attempt() -> np.ndarray | None:
    body_h = int(RNG.integers(*BODY_HEIGHT_RANGE))
    body_w = int(RNG.integers(*BODY_WIDTH_RANGE))
    count = int(RNG.integers(*OBJECT_COUNT_RANGE))

    objects: list[dict] = []
    try:
        for sym_type in _balanced_types(count):
            if sym_type == "horizontal":
                mask = _generate_horizontal_only_mask()
            elif sym_type == "vertical":
                mask = _generate_vertical_only_mask()
            else:
                mask = _generate_asymmetric_mask()
            fp = _footprint(mask, sym_type)
            objects.append({"mask": mask, "sym_type": sym_type, "fp": fp})
    except RuntimeError:
        return None

    placed = _place_objects(objects, body_h, body_w)
    if placed is None:
        return None

    grid = np.full((body_h + LEGEND_HEIGHT, body_w), ORANGE, dtype=int)
    _draw_legend(grid)
    body = grid[LEGEND_HEIGHT:, :]
    for obj in placed:
        _paste_mask(body, obj["mask"], obj["row"], obj["col"], GREY)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != (body_h, body_w):
        return None
    if not {RED, BLUE, GREEN}.issubset(set(map(int, np.unique(out)))):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
