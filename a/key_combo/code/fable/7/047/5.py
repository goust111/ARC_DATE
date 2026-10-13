import random
from collections import Counter, deque

import numpy as np


BG = 0
PALETTE = list(range(1, 10))
GEN_RETRIES = 300
MAX_OBJECTS = 4
MAX_OBJECT_AREA = 12
MAX_OBJECT_HEIGHT = 5


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return components


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in component["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _is_u_container(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    mask = _component_mask(component)
    expected = np.zeros_like(mask)
    expected[:, 0] = True
    expected[:, -1] = True
    expected[-1, :] = True
    return np.array_equal(mask, expected)


def _fill_order(container: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = container["bbox"]
    cells = []
    for r in range(r1 - 1, r0 - 1, -1):
        for c in range(c0 + 1, c1):
            cells.append((r, c))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each hanging object into the unique U container beneath its center column, then convert its area into bottom-up fill.

    Input structure:
    - The background grid contains several separated one-cell-thick monochrome U containers near the bottom.
    - Above them are one or more separated monochrome solid objects with arbitrary connected shapes.
    - Every object's bounding-box center column lies inside exactly one container's interior column span.

    Rule:
    - Erase every hanging object from its original position.
    - Assign each object to the unique container whose interior span contains that object's center column.
    - Inside each container, ignore the object's shape and keep only its area; fill that many interior cells using the object's color.
    - Filling proceeds from the container's bottom interior row upward, and within each row from left to right.

    Key insight:
    - The falling object evaporates into a color-preserving volume: geometry disappears, while pixel count and vertical-to-container pairing remain.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    components = _find_components(g, bg)
    containers = []
    objects = []
    for component in components:
        if _is_u_container(component):
            containers.append(component)
        else:
            objects.append(component)

    assert containers, "expected at least one U container"
    assert objects, "expected at least one hanging object"

    out = g.copy()
    for obj in objects:
        for r, c in obj["coords"]:
            out[r, c] = bg

    assigned: dict[int, list[dict]] = {i: [] for i in range(len(containers))}
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        assert (c1 - c0) % 2 == 0, "object bbox must have a single center column"
        center_col = (c0 + c1) // 2
        matches = []
        for idx, container in enumerate(containers):
            cr0, cc0, cr1, cc1 = container["bbox"]
            if r1 < cr0 and cc0 + 1 <= center_col <= cc1 - 1:
                matches.append(idx)
        assert len(matches) == 1, "each object must map to exactly one container"
        assigned[matches[0]].append(obj)

    for idx, container in enumerate(containers):
        fill_cells = _fill_order(container)
        pos = 0
        for obj in sorted(assigned[idx], key=lambda item: (item["bbox"][2], item["bbox"][0]), reverse=True):
            area = int(obj["area"])
            assert pos + area <= len(fill_cells), "object area exceeds container capacity"
            color = int(obj["color"])
            for r, c in fill_cells[pos : pos + area]:
                out[r, c] = color
            pos += area

    return out


def _mask_has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    canvas = np.pad(~mask, 1, constant_values=True)
    seen = np.zeros_like(canvas, dtype=bool)
    q = deque([(0, 0)])
    seen[0, 0] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h + 2 and 0 <= nc < w + 2 and not seen[nr, nc] and canvas[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return bool(np.any(canvas & ~seen))


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _random_shape(area: int, max_h: int, max_w: int) -> np.ndarray | None:
    max_w = max_w if max_w % 2 == 1 else max_w - 1
    if area < 1 or max_h < 1 or max_w < 1 or area > max_h * max_w:
        return None
    for _ in range(500):
        cells = {(0, 0)}
        while len(cells) < area:
            options = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in cells:
                        continue
                    test = cells | {(nr, nc)}
                    rows = [rr for rr, _ in test]
                    cols = [cc for _, cc in test]
                    if max(rows) - min(rows) + 1 <= max_h and max(cols) - min(cols) + 1 <= max_w:
                        options.append((nr, nc))
            if not options:
                break
            cells.add(random.choice(options))
        if len(cells) != area:
            continue
        mask = _cells_to_mask(cells)
        if mask.shape[1] % 2 == 0:
            continue
        if _mask_has_hole(mask):
            continue
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        return mask.copy()
    return None


def _draw_u(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r1, c0 : c1 + 1] = color


def _candidate_centers(container: dict, shape_w: int) -> list[int]:
    _, c0, _, c1 = container["bbox"]
    offset = shape_w // 2
    centers = []
    for center in range(c0 + 1, c1):
        left = center - offset
        right = left + shape_w - 1
        if c0 <= left and right <= c1:
            centers.append(center)
    return centers


def _build_containers() -> tuple[np.ndarray, list[dict]] | None:
    count = random.choices([2, 3, 4], weights=[3, 5, 2], k=1)[0]
    colors = random.sample(PALETTE, count)
    specs = []
    for idx in range(count):
        inner_w_max = 6 if count == 2 else 5 if count == 3 else 4
        inner_w = random.randint(2, inner_w_max)
        height = random.randint(3, 7)
        specs.append({"inner_w": inner_w, "height": height, "color": colors[idx]})

    gaps = [random.randint(1, 3) for _ in range(count - 1)]
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    width = left_margin + right_margin + sum(spec["inner_w"] + 2 for spec in specs) + sum(gaps)
    if width > 30:
        return None

    height = random.randint(max(16, max(spec["height"] for spec in specs) + 8), 28)
    grid = np.full((height, width), BG, dtype=int)
    containers = []
    col = left_margin
    for idx, spec in enumerate(specs):
        outer_w = spec["inner_w"] + 2
        r1 = height - 1
        r0 = r1 - spec["height"] + 1
        c0 = col
        c1 = c0 + outer_w - 1
        bbox = (r0, c0, r1, c1)
        containers.append(
            {
                "bbox": bbox,
                "color": spec["color"],
                "capacity": (spec["height"] - 1) * spec["inner_w"],
                "outer_w": outer_w,
            }
        )
        _draw_u(grid, bbox, spec["color"])
        if idx < len(gaps):
            col = c1 + 1 + gaps[idx]
    return grid, containers


def _assignment_counts(containers: list[dict]) -> dict[int, int] | None:
    n = len(containers)
    active_count = random.randint(2, n)
    active = random.sample(range(n), active_count)
    counts = {idx: 0 for idx in range(n)}
    for idx in active:
        counts[idx] = 1
    eligible_extra = [idx for idx in active if containers[idx]["capacity"] >= 4]
    if eligible_extra and active_count < MAX_OBJECTS and random.random() < 0.4:
        counts[random.choice(eligible_extra)] += 1
    if sum(counts.values()) < 2:
        return None
    return counts


def _choose_areas(containers: list[dict], counts: dict[int, int]) -> dict[int, list[int]] | None:
    used = set()
    chosen: dict[int, list[int]] = {idx: [] for idx in range(len(containers))}
    for idx, count in counts.items():
        if count == 0:
            continue
        container = containers[idx]
        max_shape_w = container["outer_w"] if container["outer_w"] % 2 == 1 else container["outer_w"] - 1
        area_cap = min(MAX_OBJECT_AREA, container["capacity"], MAX_OBJECT_HEIGHT * max_shape_w)
        if count == 1:
            candidates = [a for a in range(1, area_cap + 1) if a not in used]
            if not candidates:
                return None
            area = random.choice(candidates)
            chosen[idx] = [area]
            used.add(area)
            continue
        pairs = []
        for a in range(1, area_cap + 1):
            if a in used:
                continue
            for b in range(1, area_cap + 1):
                if b == a or b in used:
                    continue
                if a + b <= container["capacity"]:
                    pairs.append((a, b))
        if not pairs:
            return None
        pair = random.choice(pairs)
        chosen[idx] = [pair[0], pair[1]]
        used.add(pair[0])
        used.add(pair[1])
    return chosen


def _object_color(container_color: int, used_colors: set[int]) -> int:
    if random.random() < 0.25:
        return container_color
    candidates = [c for c in PALETTE if c != container_color and c not in used_colors]
    if not candidates:
        candidates = [c for c in PALETTE if c != container_color]
    return random.choice(candidates)


def _place_objects(grid: np.ndarray, containers: list[dict], areas: dict[int, list[int]]) -> bool:
    used_object_colors: set[int] = set()
    for idx, container in enumerate(containers):
        object_areas = areas[idx]
        if not object_areas:
            continue
        lane_bottom = container["bbox"][0] - 2 - random.randint(0, 1)
        for area in object_areas:
            shape = _random_shape(area, MAX_OBJECT_HEIGHT, container["outer_w"])
            if shape is None:
                return False
            sh, sw = shape.shape
            centers = _candidate_centers(container, sw)
            if not centers:
                return False
            center_col = random.choice(centers)
            top = lane_bottom - sh + 1
            if top < 0:
                return False
            left = center_col - sw // 2
            color = _object_color(int(container["color"]), used_object_colors)
            used_object_colors.add(color)
            for r in range(sh):
                for c in range(sw):
                    if shape[r, c]:
                        rr = top + r
                        cc = left + c
                        if grid[rr, cc] != BG:
                            return False
                        grid[rr, cc] = color
            lane_bottom = top - random.randint(2, 3)
    return True


def _gen_attempt() -> np.ndarray | None:
    built = _build_containers()
    if built is None:
        return None
    grid, containers = built
    counts = _assignment_counts(containers)
    if counts is None:
        return None
    areas = _choose_areas(containers, counts)
    if areas is None:
        return None
    if not _place_objects(grid, containers, areas):
        return None
    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
