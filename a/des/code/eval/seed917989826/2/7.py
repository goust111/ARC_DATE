from __future__ import annotations

import numpy as np


GREY = 5
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
COLOR_POOL = tuple(range(10))

MIN_CANVAS = 18
MAX_CANVAS = 34
MAX_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 300

MIN_CONTAINERS = 3
MAX_CONTAINERS = 6
MIN_INTERIOR_H = 2
MAX_INTERIOR_H = 5
MIN_INTERIOR_W = 2
MAX_INTERIOR_W = 6

AREA_TO_DIMS: dict[int, list[tuple[int, int]]] = {}
for _ih in range(MIN_INTERIOR_H, MAX_INTERIOR_H + 1):
    for _iw in range(MIN_INTERIOR_W, MAX_INTERIOR_W + 1):
        AREA_TO_DIMS.setdefault(_ih * _iw, []).append((_ih, _iw))


def _bg_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _component_bfs(
    grid: np.ndarray, bg: int
) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(grid[r0, c0]) == bg:
                continue
            color = int(grid[r0, c0])
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for dr, dc in DIR4:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and int(grid[rr, cc]) == color:
                        seen[rr, cc] = True
                        stack.append((rr, cc))

            rows = [rc[0] for rc in coords]
            cols = [rc[1] for rc in coords]
            bbox = (min(rows), min(cols), max(rows), max(cols))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "area": len(coords),
                }
            )

    return comps


def _is_rectangular_frame(comp: dict[str, object], grid: np.ndarray, bg: int) -> bool:
    r0, c0, r1, c1 = comp["bbox"]  # type: ignore[misc]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False

    perimeter = 2 * (h + w) - 4
    coords = set(comp["coords"])  # type: ignore[arg-type]
    if len(coords) != perimeter:
        return False

    for rr in range(int(r0), int(r1) + 1):
        for cc in range(int(c0), int(c1) + 1):
            on_border = rr in (int(r0), int(r1)) or cc in (int(c0), int(c1))
            present = (rr, cc) in coords
            if on_border != present:
                return False

    interior = grid[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
    return bool(interior.size) and np.all(interior == bg)


def _parse(grid: np.ndarray) -> tuple[int, list[int], list[dict[str, object]], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 4 and w >= 4

    bg = _bg_color(g)
    priority_cols = [int(c) for c in np.where(g[-1] != bg)[0]]
    assert priority_cols, "missing bottom priority row"
    priority_colors = [int(g[-1, c]) for c in priority_cols]
    assert len(set(priority_colors)) == len(priority_colors), "priority colors must be unique"

    work = g.copy()
    work[-1, :] = bg
    comps = _component_bfs(work, bg)

    containers: list[dict[str, object]] = []
    objects: list[dict[str, object]] = []
    for comp in comps:
        if _is_rectangular_frame(comp, g, bg):
            r0, c0, r1, c1 = comp["bbox"]  # type: ignore[misc]
            comp = dict(comp)
            comp["inner_area"] = int((r1 - r0 - 1) * (c1 - c0 - 1))
            containers.append(comp)
        else:
            objects.append(comp)

    assert containers, "no containers found"
    assert objects, "no objects found"
    assert len(containers) == len(objects), "object/container count mismatch"

    object_colors = {int(obj["color"]) for obj in objects}
    assert object_colors == set(priority_colors), "priority row must list object colors exactly"
    return bg, priority_colors, containers, objects


def _container_sort_key(comp: dict[str, object]) -> tuple[int, int, int, int, int]:
    r0, c0, r1, c1 = comp["bbox"]  # type: ignore[misc]
    return (
        int(comp["inner_area"]),
        int(r0),
        int(c0),
        int(r1 - r0 + 1),
        int(c1 - c0 + 1),
    )


def _object_sort_key(
    comp: dict[str, object], priority_rank: dict[int, int]
) -> tuple[int, int, int, int]:
    r0, c0, _r1, _c1 = comp["bbox"]  # type: ignore[misc]
    return (
        int(comp["area"]),
        int(priority_rank[int(comp["color"])]),
        int(r0),
        int(c0),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each solid colored object into a hollow rectangular container with the same area.

    Containers are sorted by interior area. Objects are sorted by area, and equal-area
    objects are ordered by the bottom-row color sequence. Equal-area containers, if any,
    are consumed in reading order.
    """
    g = np.asarray(grid, dtype=int)
    bg, priority_colors, containers, objects = _parse(g)
    priority_rank = {int(color): idx for idx, color in enumerate(priority_colors)}

    ordered_containers = sorted(containers, key=_container_sort_key)
    ordered_objects = sorted(objects, key=lambda comp: _object_sort_key(comp, priority_rank))

    out = g.copy()
    for obj in objects:
        for r, c in obj["coords"]:  # type: ignore[arg-type]
            out[int(r), int(c)] = bg

    for container, obj in zip(ordered_containers, ordered_objects, strict=True):
        assert int(container["inner_area"]) == int(obj["area"]), "area mismatch between object and container"
        r0, c0, r1, c1 = container["bbox"]  # type: ignore[misc]
        out[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)] = int(obj["color"])

    return out


def _group_patterns(n: int) -> list[list[int]]:
    patterns: dict[int, list[list[int]]] = {
        3: [[2, 1], [3]],
        4: [[2, 1, 1], [2, 2], [3, 1]],
        5: [[2, 1, 1, 1], [2, 2, 1], [3, 1, 1], [3, 2]],
        6: [[2, 1, 1, 1, 1], [2, 2, 1, 1], [2, 2, 2], [3, 1, 1, 1], [3, 2, 1], [3, 3]],
    }
    return patterns[n]


def _sample_area_multiset(
    rng: np.random.Generator, n: int, force_duplicates: bool
) -> list[int]:
    if not force_duplicates:
        areas = rng.choice(np.array(sorted(AREA_TO_DIMS)), size=n, replace=False)
        return [int(x) for x in areas]

    sizes = list(_group_patterns(n)[int(rng.integers(0, len(_group_patterns(n))))])
    rng.shuffle(sizes)

    available = set(AREA_TO_DIMS)
    out: list[int] = []
    for size in sizes:
        candidates = [area for area in available if len(AREA_TO_DIMS[area]) >= size]
        if not candidates:
            return []
        area = int(candidates[int(rng.integers(0, len(candidates)))])
        available.remove(area)
        out.extend([area] * size)
    rng.shuffle(out)
    return out


def _random_polyomino(
    rng: np.random.Generator, area: int, color: int, bg: int
) -> np.ndarray:
    dims: list[tuple[int, int]] = []
    max_h = min(6, area)
    max_w = min(7, area)
    for h in range(1, max_h + 1):
        for w in range(1, max_w + 1):
            if h * w < area:
                continue
            slack = h * w - area
            if slack > max(4, area // 2):
                continue
            dims.append((h, w))

    rich_dims = [dim for dim in dims if dim[0] > 1 and dim[1] > 1]
    if rich_dims and rng.random() < 0.8:
        dims = rich_dims
    h, w = dims[int(rng.integers(0, len(dims)))]

    seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {seed}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        seen_frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in DIR4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells and (rr, cc) not in seen_frontier:
                    frontier.append((rr, cc))
                    seen_frontier.add((rr, cc))
        pick = frontier[int(rng.integers(0, len(frontier)))]
        cells.add(pick)

    rows = [rc[0] for rc in cells]
    cols = [rc[1] for rc in cells]
    r0 = min(rows)
    c0 = min(cols)
    r1 = max(rows)
    c1 = max(cols)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in cells:
        patch[r - r0, c - c0] = color
    return patch


def _container_patch(interior_h: int, interior_w: int, bg: int) -> np.ndarray:
    patch = np.full((interior_h + 2, interior_w + 2), bg, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    return patch


def _place_patch(
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    transparent: int,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    ph, pw = patch.shape
    usable_h = grid.shape[0] - 2
    usable_w = grid.shape[1]
    max_r = usable_h - ph
    max_c = usable_w - pw
    if max_r < 0 or max_c < 0:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = int(rng.integers(0, max_r + 1))
        c0 = int(rng.integers(0, max_c + 1))
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(grid.shape[0], r0 + ph + 1)
        cc1 = min(grid.shape[1], c0 + pw + 1)
        if occupied[rr0:rr1, cc0:cc1].any():
            continue
        mask = patch != transparent
        window = grid[r0 : r0 + ph, c0 : c0 + pw]
        window[mask] = patch[mask]
        occupied[rr0:rr1, cc0:cc1] = True
        return r0, c0

    return None


def _expected_from_records(
    inp: np.ndarray,
    bg: int,
    priority_colors: list[int],
    containers: list[dict[str, object]],
    objects: list[dict[str, object]],
) -> np.ndarray:
    out = inp.copy()
    for obj in objects:
        for r, c in obj["coords"]:  # type: ignore[arg-type]
            out[int(r), int(c)] = bg

    priority_rank = {int(color): idx for idx, color in enumerate(priority_colors)}
    ordered_containers = sorted(containers, key=_container_sort_key)
    ordered_objects = sorted(objects, key=lambda comp: _object_sort_key(comp, priority_rank))

    for container, obj in zip(ordered_containers, ordered_objects, strict=True):
        assert int(container["inner_area"]) == int(obj["area"])
        r0, c0, r1, c1 = container["bbox"]  # type: ignore[misc]
        out[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)] = int(obj["color"])

    return out


def _same_area_needs_priority(
    objects: list[dict[str, object]], priority_colors: list[int]
) -> bool:
    priority_rank = {int(color): idx for idx, color in enumerate(priority_colors)}
    by_area: dict[int, list[dict[str, object]]] = {}
    for obj in objects:
        by_area.setdefault(int(obj["area"]), []).append(obj)

    for area, group in by_area.items():
        if len(group) < 2:
            continue
        row_major = [int(obj["color"]) for obj in sorted(group, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))]  # type: ignore[index]
        by_priority = [
            int(obj["color"])
            for obj in sorted(group, key=lambda obj: (priority_rank[int(obj["color"])], obj["bbox"][0], obj["bbox"][1]))
        ]
        if row_major != by_priority:
            return True
    return False


def _generate_case(rng: np.random.Generator) -> np.ndarray | None:
    bg_choices = [color for color in COLOR_POOL if color != GREY]
    bg = int(bg_choices[int(rng.integers(0, len(bg_choices)))])

    n = int(rng.integers(MIN_CONTAINERS, MAX_CONTAINERS + 1))
    force_duplicates = bool(rng.integers(0, 2))
    areas = _sample_area_multiset(rng, n, force_duplicates)
    if not areas:
        return None

    available_colors = [color for color in COLOR_POOL if color not in (bg, GREY)]
    obj_colors = [int(x) for x in rng.choice(np.array(available_colors), size=n, replace=False)]

    container_patches: list[tuple[np.ndarray, int]] = []
    for area in areas:
        dims = AREA_TO_DIMS[area]
        interior_h, interior_w = dims[int(rng.integers(0, len(dims)))]
        container_patches.append((_container_patch(interior_h, interior_w, bg), int(area)))

    object_patches: list[tuple[np.ndarray, int, int]] = []
    for color, area in zip(obj_colors, areas, strict=True):
        object_patches.append((_random_polyomino(rng, int(area), int(color), bg), int(color), int(area)))

    max_patch_h = max(max(patch.shape[0] for patch, _area in container_patches), max(patch.shape[0] for patch, _color, _area in object_patches))
    max_patch_w = max(max(patch.shape[1] for patch, _area in container_patches), max(patch.shape[1] for patch, _color, _area in object_patches))
    footprint = sum((patch.shape[0] + 2) * (patch.shape[1] + 2) for patch, _area in container_patches)
    footprint += sum((patch.shape[0] + 2) * (patch.shape[1] + 2) for patch, _color, _area in object_patches)

    h_low = max(MIN_CANVAS, max_patch_h + 4)
    w_low = max(MIN_CANVAS, max(max_patch_w + 2, n + 2))

    for _ in range(40):
        h = int(rng.integers(h_low, MAX_CANVAS + 1))
        w = int(rng.integers(w_low, MAX_CANVAS + 1))
        if h * w >= footprint + 80:
            break
    else:
        return None

    inp = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    occupied[h - 2 :, :] = True

    container_records: list[dict[str, object]] = []
    object_records: list[dict[str, object]] = []

    container_order = list(range(len(container_patches)))
    object_order = list(range(len(object_patches)))
    rng.shuffle(container_order)
    rng.shuffle(object_order)
    container_order.sort(key=lambda idx: container_patches[idx][0].shape[0] * container_patches[idx][0].shape[1], reverse=True)
    object_order.sort(key=lambda idx: object_patches[idx][0].shape[0] * object_patches[idx][0].shape[1], reverse=True)

    for idx in container_order:
        patch, area = container_patches[idx]
        loc = _place_patch(inp, occupied, patch, bg, rng)
        if loc is None:
            return None
        r0, c0 = loc
        r1 = r0 + patch.shape[0] - 1
        c1 = c0 + patch.shape[1] - 1
        coords = [(r0 + rr, c0 + cc) for rr, cc in zip(*np.where(patch == GREY), strict=False)]
        container_records.append(
            {
                "color": GREY,
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "area": len(coords),
                "inner_area": area,
            }
        )

    for idx in object_order:
        patch, color, area = object_patches[idx]
        loc = _place_patch(inp, occupied, patch, bg, rng)
        if loc is None:
            return None
        r0, c0 = loc
        rows, cols = np.where(patch == color)
        coords = [(r0 + int(rr), c0 + int(cc)) for rr, cc in zip(rows, cols, strict=False)]
        bbox = (
            min(rc[0] for rc in coords),
            min(rc[1] for rc in coords),
            max(rc[0] for rc in coords),
            max(rc[1] for rc in coords),
        )
        object_records.append(
            {
                "color": color,
                "coords": coords,
                "bbox": bbox,
                "area": area,
            }
        )

    priority_colors = obj_colors[:]
    rng.shuffle(priority_colors)
    start = int(rng.integers(0, w - n + 1))
    inp[-1, start : start + n] = np.array(priority_colors, dtype=int)

    if force_duplicates and not _same_area_needs_priority(object_records, priority_colors):
        return None

    vals, cnts = np.unique(inp, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None
    if list(cnts).count(int(cnts.max())) != 1:
        return None

    expected = _expected_from_records(inp, bg, priority_colors, container_records, object_records)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _generate_case(rng)
        if case is not None:
            return case
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
