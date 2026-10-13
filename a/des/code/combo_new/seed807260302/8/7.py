import random

import numpy as np


BACKGROUND = 0
WALL_COLOR = 1
CONTENT_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)
OPENINGS = ("up", "down", "left", "right")

GRID_MIN = 12
GRID_MAX = 30
CONTAINER_COUNT_RANGE = (2, 4)
BOX_H_RANGE = (4, 8)
BOX_W_RANGE = (4, 8)
PLACEMENT_MARGIN = 1
PLACEMENT_GAP = 1
PLACEMENT_ATTEMPTS = 80
GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    return int(np.bincount(grid.ravel()).argmax())


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _container_cells(bbox: tuple[int, int, int, int], opening: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    if opening != "up":
        cells.update((r0, c) for c in range(c0, c1 + 1))
    if opening != "down":
        cells.update((r1, c) for c in range(c0, c1 + 1))
    if opening != "left":
        cells.update((r, c0) for r in range(r0, r1 + 1))
    if opening != "right":
        cells.update((r, c1) for r in range(r0, r1 + 1))
    return cells


def _interior_cells(info: dict[str, object]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = info["bbox"]
    opening = info["opening"]
    if opening == "up":
        rows = range(r0, r1)
        cols = range(c0 + 1, c1)
    elif opening == "down":
        rows = range(r0 + 1, r1 + 1)
        cols = range(c0 + 1, c1)
    elif opening == "left":
        rows = range(r0 + 1, r1)
        cols = range(c0, c1)
    else:
        rows = range(r0 + 1, r1)
        cols = range(c0 + 1, c1 + 1)
    return [(r, c) for r in rows for c in cols]


def _packing_order(info: dict[str, object]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = info["bbox"]
    opening = info["opening"]
    order: list[tuple[int, int]] = []
    if opening == "up":
        for r in range(r1 - 1, r0 - 1, -1):
            for c in range(c0 + 1, c1):
                order.append((r, c))
    elif opening == "down":
        for r in range(r0 + 1, r1 + 1):
            for c in range(c0 + 1, c1):
                order.append((r, c))
    elif opening == "left":
        for c in range(c1 - 1, c0 - 1, -1):
            for r in range(r0 + 1, r1):
                order.append((r, c))
    else:
        for c in range(c0 + 1, c1 + 1):
            for r in range(r0 + 1, r1):
                order.append((r, c))
    return order


def _match_container(comp: list[tuple[int, int]]) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(comp)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    cells = set(comp)
    matches = [opening for opening in OPENINGS if cells == _container_cells((r0, c0, r1, c1), opening)]
    if len(matches) != 1:
        return None
    opening = matches[0]
    info = {"bbox": (r0, c0, r1, c1), "opening": opening}
    if not _interior_cells(info):
        return None
    return info


def _find_containers(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    containers: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            info = _match_container(comp)
            if info is not None:
                info["wall_color"] = color
                containers.append(info)
    containers.sort(key=lambda item: item["bbox"])
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gather each container's scattered interior pixels into a compact block resting against that container's own bottom wall.

    Input structure:
    - The background is the dominant color.
    - Several one-pixel-thick U- or C-shaped containers appear as 4-connected components, each formed by exactly three sides of an axis-aligned rectangle and open on one side.
    - Each container encloses a rectangular interior region, and the interior contains zero or more scattered pixels of a single non-background color.

    Rule:
    - Detect every three-sided rectangular container and determine which side is open.
    - Define the container's bottom as the wall opposite the opening, count the interior colored pixels, and erase their original positions.
    - Repack those pixels into a tight rectangle anchored to the bottom wall: fill the row or column adjacent to the bottom wall first across its full span, then continue layer by layer toward the opening in row-major order.
    - Leave the container walls and the outer background unchanged.

    Key insight:
    - "Down" is local to each container: the missing side determines the opening, so the pixels must be packed relative to that container's own orientation rather than global screen-down.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()

    for info in _find_containers(g, bg):
        interior = _interior_cells(info)
        payload = [int(g[r, c]) for r, c in interior if int(g[r, c]) != bg]
        if not payload:
            continue
        colors = set(payload)
        assert len(colors) == 1, "each container interior must use a single payload color"
        fill_color = next(iter(colors))

        for r, c in interior:
            out[r, c] = bg

        order = _packing_order(info)
        count = len(payload)
        assert count <= len(order), "payload must fit inside the interior"
        for r, c in order[:count]:
            out[r, c] = fill_color

    return out


def _random_bbox(opening: str, grid_h: int, grid_w: int) -> tuple[int, int, int, int] | None:
    if opening in ("up", "down"):
        height = random.randint(4, BOX_H_RANGE[1])
        width = random.randint(4, BOX_W_RANGE[1] + 1)
    else:
        height = random.randint(4, BOX_H_RANGE[1] + 1)
        width = random.randint(4, BOX_W_RANGE[1])
    if height + 2 * PLACEMENT_MARGIN > grid_h or width + 2 * PLACEMENT_MARGIN > grid_w:
        return None
    r0 = random.randint(PLACEMENT_MARGIN, grid_h - height - PLACEMENT_MARGIN)
    c0 = random.randint(PLACEMENT_MARGIN, grid_w - width - PLACEMENT_MARGIN)
    return r0, c0, r0 + height - 1, c0 + width - 1


def _rectangles_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _line_length(info: dict[str, object]) -> int:
    r0, c0, r1, c1 = info["bbox"]
    if info["opening"] in ("up", "down"):
        return c1 - c0 - 1
    return r1 - r0 - 1


def _choose_payload_count(info: dict[str, object]) -> int:
    area = len(_interior_cells(info))
    span = _line_length(info)
    assert area >= 2 and span >= 1
    full_rows = [k * span for k in range(1, area // span + 1) if k * span < area]
    partial_rows = [n for n in range(1, area) if n % span != 0]
    if full_rows and partial_rows:
        pool = full_rows if random.random() < 0.5 else partial_rows
        return random.choice(pool)
    return random.randint(1, area - 1)


def _random_scatter(interior: list[tuple[int, int]], packed_prefix: set[tuple[int, int]], count: int) -> set[tuple[int, int]] | None:
    for _ in range(40):
        scatter = set(random.sample(interior, count))
        if scatter != packed_prefix:
            return scatter
    return None


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(*CONTAINER_COUNT_RANGE)
    min_side = max(GRID_MIN, 10 + 4 * (n_containers - 1))
    grid_h = random.randint(min_side, GRID_MAX)
    grid_w = random.randint(min_side, GRID_MAX)
    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)

    directions = random.sample(OPENINGS, n_containers)
    payload_colors = random.sample(CONTENT_COLORS, n_containers)
    placed_boxes: list[tuple[int, int, int, int]] = []
    infos: list[dict[str, object]] = []

    for opening, payload_color in zip(directions, payload_colors):
        bbox = None
        info = None
        for _ in range(PLACEMENT_ATTEMPTS):
            candidate = _random_bbox(opening, grid_h, grid_w)
            if candidate is None:
                continue
            if any(_rectangles_too_close(candidate, other, PLACEMENT_GAP) for other in placed_boxes):
                continue
            candidate_info = {"bbox": candidate, "opening": opening, "wall_color": WALL_COLOR, "payload_color": payload_color}
            if len(_interior_cells(candidate_info)) < 4:
                continue
            bbox = candidate
            info = candidate_info
            break
        if bbox is None or info is None:
            return None
        placed_boxes.append(bbox)
        infos.append(info)

    for info in infos:
        for r, c in _container_cells(info["bbox"], info["opening"]):
            grid[r, c] = WALL_COLOR

        interior = _interior_cells(info)
        order = _packing_order(info)
        count = _choose_payload_count(info)
        packed_prefix = set(order[:count])
        scatter = _random_scatter(interior, packed_prefix, count)
        if scatter is None:
            return None
        for r, c in scatter:
            grid[r, c] = info["payload_color"]

    containers = _find_containers(grid, BACKGROUND)
    if len(containers) != n_containers:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid container-packing input")
