import random
from collections import Counter, deque

import numpy as np


ORIENTATIONS = ("up", "down", "left", "right")


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _u_patch(height: int, width: int, orientation: str) -> np.ndarray:
    patch = np.zeros((height, width), dtype=bool)
    if orientation == "up":
        patch[:, 0] = True
        patch[:, -1] = True
        patch[-1, :] = True
    elif orientation == "down":
        patch[:, 0] = True
        patch[:, -1] = True
        patch[0, :] = True
    elif orientation == "left":
        patch[0, :] = True
        patch[-1, :] = True
        patch[:, -1] = True
    elif orientation == "right":
        patch[0, :] = True
        patch[-1, :] = True
        patch[:, 0] = True
    else:
        raise ValueError(f"unknown orientation: {orientation}")
    return patch


def _parse_container(component: list[tuple[int, int]]) -> dict | None:
    rows = [r for r, _ in component]
    cols = [c for _, c in component]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    patch = np.zeros((height, width), dtype=bool)
    for r, c in component:
        patch[r - r0, c - c0] = True

    matches = [orientation for orientation in ORIENTATIONS if np.array_equal(patch, _u_patch(height, width, orientation))]
    if len(matches) != 1:
        return None

    orientation = matches[0]
    if orientation in ("up", "down"):
        span = width - 2
        depth = height - 1
    else:
        span = height - 2
        depth = width - 1
    return {
        "orientation": orientation,
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "height": height,
        "width": width,
        "span": span,
        "depth": depth,
    }


def _detect_grey_color(grid: np.ndarray, bg: int) -> int:
    best_color = None
    best_key = None
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        comps = _components(grid == color)
        max_area = max((len(comp) for comp in comps), default=0)
        key = (max_area, int(np.count_nonzero(grid == color)))
        if best_key is None or key > best_key:
            best_key = key
            best_color = color
    assert best_color is not None
    return int(best_color)


def _detect_containers(grid: np.ndarray) -> tuple[int, int, list[dict]]:
    bg = _most_common_color(grid)
    grey = _detect_grey_color(grid, bg)
    containers: list[dict] = []
    for component in _components(grid == grey):
        container = _parse_container(component)
        assert container is not None, "grey components must all be U-shaped containers"
        containers.append(container)
    containers.sort(key=lambda item: (item["r0"], item["c0"], item["orientation"]))
    assert containers, "expected at least one container"
    return bg, grey, containers


def _seed_strip_cells(shape: tuple[int, int], container: dict) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = container["r0"], container["c0"], container["r1"], container["c1"]
    orientation = container["orientation"]
    cells: list[tuple[int, int]] = []
    if orientation == "up":
        for r in range(0, r0):
            for c in range(c0 + 1, c1):
                cells.append((r, c))
    elif orientation == "down":
        for r in range(r1 + 1, h):
            for c in range(c0 + 1, c1):
                cells.append((r, c))
    elif orientation == "left":
        for r in range(r0 + 1, r1):
            for c in range(0, c0):
                cells.append((r, c))
    elif orientation == "right":
        for r in range(r0 + 1, r1):
            for c in range(c1 + 1, w):
                cells.append((r, c))
    else:
        raise ValueError(f"unknown orientation: {orientation}")
    return cells


def _collect_seeds(grid: np.ndarray, bg: int, grey: int, container: dict) -> tuple[int | None, list[tuple[int, int]]]:
    seed_positions: list[tuple[int, int]] = []
    colors = set()
    for r, c in _seed_strip_cells(grid.shape, container):
        value = int(grid[r, c])
        if value in (bg, grey):
            continue
        seed_positions.append((r, c))
        colors.add(value)
    assert len(colors) <= 1, "each container should see at most one seed color"
    seed_color = next(iter(colors)) if colors else None
    return seed_color, seed_positions


def _fill_layers(out: np.ndarray, container: dict, color: int, layers: int) -> None:
    r0, c0, r1, c1 = container["r0"], container["c0"], container["r1"], container["c1"]
    orientation = container["orientation"]
    depth = container["depth"]
    layers = min(int(layers), int(depth))
    if layers <= 0:
        return

    if orientation == "up":
        for offset in range(layers):
            out[r1 - 1 - offset, c0 + 1 : c1] = color
    elif orientation == "down":
        for offset in range(layers):
            out[r0 + 1 + offset, c0 + 1 : c1] = color
    elif orientation == "left":
        for offset in range(layers):
            out[r0 + 1 : r1, c1 - 1 - offset] = color
    elif orientation == "right":
        for offset in range(layers):
            out[r0 + 1 : r1, c0 + 1 + offset] = color
    else:
        raise ValueError(f"unknown orientation: {orientation}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push scattered seeds into U-shaped containers and fill them layer by layer from the closed side.
    """
    g = np.asarray(grid, dtype=int)
    bg, grey, containers = _detect_containers(g)
    out = g.copy()
    for container in containers:
        seed_color, seed_positions = _collect_seeds(g, bg, grey, container)
        for r, c in seed_positions:
            out[r, c] = bg
        if seed_color is not None:
            _fill_layers(out, container, seed_color, len(seed_positions))
    return out


GEN_RETRIES = 400
GRID_HEIGHT_RANGE = (14, 24)
GRID_WIDTH_RANGE = (14, 24)
CONTAINER_COUNT_OPTIONS = (1, 2, 3)
CONTAINER_COUNT_WEIGHTS = (3, 4, 2)
CONTAINER_HEIGHT_RANGE = (3, 7)
CONTAINER_WIDTH_RANGE = (3, 7)
MAX_NEAR_EDGE_GAP = 8
PLACEMENT_TRIES = 120


def _shape_cells(container: dict) -> set[tuple[int, int]]:
    r0, c0 = container["r0"], container["c0"]
    patch = _u_patch(container["height"], container["width"], container["orientation"])
    cells = set()
    for rr, cc in np.argwhere(patch):
        cells.add((r0 + int(rr), c0 + int(cc)))
    return cells


def _corridor_cells(shape: tuple[int, int], container: dict) -> set[tuple[int, int]]:
    return set(_seed_strip_cells(shape, container))


def _expanded_bbox_cells(shape: tuple[int, int], container: dict, margin: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    cells = set()
    for r in range(max(0, container["r0"] - margin), min(h, container["r1"] + margin + 1)):
        for c in range(max(0, container["c0"] - margin), min(w, container["c1"] + margin + 1)):
            cells.add((r, c))
    return cells


def _capacity_for_gap(gap: int, span: int) -> int:
    return (gap * span + 1) // 2


def _seed_count(depth: int, span: int, max_gap: int) -> tuple[int, int] | None:
    modes: dict[str, list[int]] = {}
    for gap in range(1, max_gap + 1):
        capacity = _capacity_for_gap(gap, span)
        if depth > 1 and capacity >= 1:
            modes.setdefault("under", []).append(gap)
        if capacity >= depth:
            modes.setdefault("exact", []).append(gap)
        if capacity >= depth + 1:
            modes.setdefault("over", []).append(gap)
    if not modes:
        return None

    mode = random.choice(list(modes))
    gap = random.choice(modes[mode])
    capacity = _capacity_for_gap(gap, span)
    if mode == "under":
        count = random.randint(1, min(depth - 1, capacity))
    elif mode == "exact":
        count = depth
    else:
        count = random.randint(depth + 1, min(depth + 3, capacity))
    return count, gap


def _sample_seed_positions(corridor: set[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    by_parity = {0: [], 1: []}
    for cell in sorted(corridor):
        by_parity[(cell[0] + cell[1]) & 1].append(cell)
    viable = [cells for cells in by_parity.values() if len(cells) >= count]
    if not viable:
        return None
    pool = random.choice(viable)
    return random.sample(pool, count)


def _sample_container(h: int, w: int, blocked: set[tuple[int, int]], seed_color: int) -> dict | None:
    orientation = random.choice(ORIENTATIONS)
    box_h = random.randint(CONTAINER_HEIGHT_RANGE[0], min(CONTAINER_HEIGHT_RANGE[1], h - 2))
    box_w = random.randint(CONTAINER_WIDTH_RANGE[0], min(CONTAINER_WIDTH_RANGE[1], w - 2))
    if orientation in ("up", "down"):
        span = box_w - 2
        depth = box_h - 1
        max_gap = min(MAX_NEAR_EDGE_GAP, h - box_h - 1)
        if max_gap < 1 or w - box_w - 2 < 1:
            return None
        count_gap = _seed_count(depth, span, max_gap)
        if count_gap is None:
            return None
        count, gap = count_gap
        c0 = random.randint(1, w - box_w - 2)
        r0 = gap if orientation == "up" else h - gap - box_h
    else:
        span = box_h - 2
        depth = box_w - 1
        max_gap = min(MAX_NEAR_EDGE_GAP, w - box_w - 1)
        if max_gap < 1 or h - box_h - 2 < 1:
            return None
        count_gap = _seed_count(depth, span, max_gap)
        if count_gap is None:
            return None
        count, gap = count_gap
        r0 = random.randint(1, h - box_h - 2)
        c0 = gap if orientation == "left" else w - gap - box_w

    container = {
        "orientation": orientation,
        "r0": r0,
        "c0": c0,
        "r1": r0 + box_h - 1,
        "c1": c0 + box_w - 1,
        "height": box_h,
        "width": box_w,
        "span": span,
        "depth": depth,
        "seed_color": seed_color,
        "seed_count": count,
    }
    corridor = _corridor_cells((h, w), container)
    block_cells = corridor | _expanded_bbox_cells((h, w), container, margin=1)
    if block_cells & blocked:
        return None
    seed_positions = _sample_seed_positions(corridor, count)
    if seed_positions is None:
        return None
    container["seed_positions"] = seed_positions
    container["shape_cells"] = _shape_cells(container)
    container["block_cells"] = block_cells
    return container


def _planned_output(grid: np.ndarray, bg: int, containers: list[dict]) -> np.ndarray:
    out = grid.copy()
    for container in containers:
        for r, c in container["seed_positions"]:
            out[r, c] = bg
        _fill_layers(out, container, int(container["seed_color"]), int(container["seed_count"]))
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    max_containers = 3 if min(height, width) >= 18 else 2
    choices = [count for count in CONTAINER_COUNT_OPTIONS if count <= max_containers]
    weights = [CONTAINER_COUNT_WEIGHTS[CONTAINER_COUNT_OPTIONS.index(count)] for count in choices]
    container_count = random.choices(choices, weights=weights, k=1)[0]

    palette = random.sample(range(10), container_count + 2)
    bg, grey = palette[0], palette[1]
    seed_colors = palette[2:]

    blocked: set[tuple[int, int]] = set()
    containers: list[dict] = []
    for seed_color in seed_colors:
        placed = None
        for _ in range(PLACEMENT_TRIES):
            placed = _sample_container(height, width, blocked, int(seed_color))
            if placed is not None:
                containers.append(placed)
                blocked |= placed["block_cells"]
                break
        if placed is None:
            return None

    grid = np.full((height, width), int(bg), dtype=int)
    for container in containers:
        for r, c in container["shape_cells"]:
            grid[r, c] = int(grey)
        for r, c in container["seed_positions"]:
            grid[r, c] = int(container["seed_color"])

    if _most_common_color(grid) != int(bg):
        return None

    bg2, grey2, detected = _detect_containers(grid)
    if bg2 != int(bg) or grey2 != int(grey) or len(detected) != len(containers):
        return None

    expected = _planned_output(grid, int(bg), containers)
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
