from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)
GENERATION_RETRIES = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _monochrome_components(grid: np.ndarray) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BLACK):
        for coords in _mask_components(grid == color):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "area": int(coords.shape[0]),
                    "bbox": _bbox(coords),
                }
            )
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touches_boundary(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords.tolist())


def _component_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in coords.tolist()}


def _center_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) // 2, (c0 + c1) // 2


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    comps = _monochrome_components(g)
    assert comps, "expected a hollow border and external objects"

    cell_to_comp: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(comps):
        for rc in comp["coords"].tolist():
            cell_to_comp[(int(rc[0]), int(rc[1]))] = idx

    cavity_choice = None
    for cavity in _mask_components(g == BLACK):
        if _touches_boundary(cavity, g.shape):
            continue
        touched: dict[int, int] = {}
        for r, c in cavity.tolist():
            for dr, dc in NEIGHBORS4:
                nr, nc = int(r) + dr, int(c) + dc
                if _in_bounds(g.shape, nr, nc) and g[nr, nc] != BLACK:
                    cid = cell_to_comp[(nr, nc)]
                    touched[cid] = touched.get(cid, 0) + 1
        if not touched:
            continue
        dominant_id = max(touched, key=lambda cid: (touched[cid], comps[cid]["area"]))
        score = (
            1 if len(touched) == 1 else 0,
            int(comps[dominant_id]["area"]),
            int(touched[dominant_id]),
            int(cavity.shape[0]),
        )
        if cavity_choice is None or score > cavity_choice[0]:
            cavity_choice = (score, dominant_id, cavity)

    assert cavity_choice is not None, "expected exactly one enclosed cavity"
    border_id = int(cavity_choice[1])
    border = comps[border_id]
    cavity = np.asarray(cavity_choice[2], dtype=int)

    objects = [comp for idx, comp in enumerate(comps) if idx != border_id]
    areas = [int(obj["area"]) for obj in objects]
    assert len(areas) == len(set(areas)), "external object areas must be unique"
    assert 2 <= len(objects) <= 4, "expected 2-4 external objects"

    return {
        "border": border,
        "cavity": cavity,
        "objects": objects,
    }


def _paint_cavity_rings(out: np.ndarray, cavity: np.ndarray, ordered_colors: list[int]) -> np.ndarray:
    bbox = _bbox(cavity)
    cr, cc = _center_from_bbox(bbox)
    n = len(ordered_colors)
    for r, c in cavity.tolist():
        dist = max(abs(int(r) - cr), abs(int(c) - cc))
        if dist < n:
            out[int(r), int(c)] = int(ordered_colors[n - 1 - dist])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the hollow sprite's cavity with area-ordered concentric square rings clipped by the cavity shape.

    Input structure:
    - Black background contains one large single-color hollow sprite whose border encloses one black cavity.
    - Outside the sprite sit 2-4 small monochrome 4-connected objects, each with a distinct color and distinct area.
    - The cavity is fully internal and its bounding box has a well-defined center.

    Rule:
    - Find the hollow sprite and the enclosed black cavity inside it.
    - Find every external non-black object other than the border and sort them by decreasing pixel count.
    - Use the cavity bounding-box center as the center of nested odd-sized square rings: the largest object gives the outer ring of side 2N-1, then 2N-3, and so on until the smallest object gives the center pixel.
    - Paint only cavity cells that lie on those rings; parts of the rings outside the cavity are discarded, while the border and the external objects stay unchanged.

    Key insight:
    The outside objects do not move; their areas define the outside-to-inside color order of a concentric square pattern that is then clipped by the hollow sprite's irregular cavity.
    """
    g = np.asarray(grid, dtype=int)
    scene = _extract_scene(g)
    ordered = sorted(scene["objects"], key=lambda obj: int(obj["area"]), reverse=True)
    ordered_colors = [int(obj["color"]) for obj in ordered]
    out = g.copy()
    return _paint_cavity_rings(out, scene["cavity"], ordered_colors)


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _transform_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    cur = set(cells)
    for _ in range(random.randrange(4)):
        cur = {(c, -r) for r, c in cur}
        cur = _normalize_cells(cur)
    if random.random() < 0.5:
        max_c = max(c for _, c in cur)
        cur = {(r, max_c - c) for r, c in cur}
    if random.random() < 0.5:
        max_r = max(r for r, _ in cur)
        cur = {(max_r - r, c) for r, c in cur}
    return _normalize_cells(cur)


def _random_blob_cells(area: int) -> set[tuple[int, int]]:
    for _ in range(80):
        max_dim = min(5, area)
        feasible = [
            (h, w)
            for h in range(2, max_dim + 1)
            for w in range(2, max_dim + 1)
            if h * w >= area
        ]
        h, w = random.choice(feasible)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = set()

        def add_frontier(r: int, c: int) -> None:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        add_frontier(*start)
        while len(cells) < area:
            if not frontier:
                break
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            add_frontier(*cell)
        if len(cells) == area:
            return _transform_cells(cells)
    raise RuntimeError("failed to grow a connected blob")


def _pattern_mask_local(shape: tuple[int, int], n: int) -> np.ndarray:
    h, w = shape
    cr, cc = h // 2, w // 2
    rad = n - 1
    out = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if max(abs(r - cr), abs(c - cc)) <= rad:
                out[r, c] = True
    return out


def _is_connected_true(mask: np.ndarray) -> bool:
    comps = _mask_components(mask)
    return len(comps) == 1


def _bite_templates(h: int, w: int, n: int) -> list[set[tuple[int, int]]]:
    cr, cc = h // 2, w // 2
    rad = n - 1
    sr0, sc0 = cr - rad, cc - rad
    sr1, sc1 = cr + rad, cc + rad
    templates: list[set[tuple[int, int]]] = []

    corners = [
        ((sr0, sc0), (1, 1)),
        ((sr0, sc1), (1, -1)),
        ((sr1, sc0), (-1, 1)),
        ((sr1, sc1), (-1, -1)),
    ]
    for (r, c), (dr, dc) in corners:
        if 0 <= r < h and 0 <= c < w:
            templates.append({(r, c)})
            if _in_bounds((h, w), r + dr, c) and _in_bounds((h, w), r, c + dc):
                templates.append({(r, c), (r + dr, c), (r, c + dc)})
            if _in_bounds((h, w), r + dr, c + dc):
                templates.append({(r, c), (r + dr, c), (r, c + dc), (r + dr, c + dc)})

    if sc1 - sc0 >= 2:
        top = {(sr0, cc - 1), (sr0, cc), (sr0, cc + 1)}
        bottom = {(sr1, cc - 1), (sr1, cc), (sr1, cc + 1)}
        templates.append(top)
        templates.append(bottom)
        if _in_bounds((h, w), sr0 + 1, cc):
            templates.append(top | {(sr0 + 1, cc)})
        if _in_bounds((h, w), sr1 - 1, cc):
            templates.append(bottom | {(sr1 - 1, cc)})

    if sr1 - sr0 >= 2:
        left = {(cr - 1, sc0), (cr, sc0), (cr + 1, sc0)}
        right = {(cr - 1, sc1), (cr, sc1), (cr + 1, sc1)}
        templates.append(left)
        templates.append(right)
        if _in_bounds((h, w), cr, sc0 + 1):
            templates.append(left | {(cr, sc0 + 1)})
        if _in_bounds((h, w), cr, sc1 - 1):
            templates.append(right | {(cr, sc1 - 1)})

    clean_templates = []
    for tpl in templates:
        if all(_in_bounds((h, w), r, c) for r, c in tpl):
            clean_templates.append(tpl)
    return clean_templates


def _sample_cavity_mask(n: int) -> np.ndarray:
    dim_choices = {
        2: [(3, 9), (9, 3), (5, 5), (5, 7), (7, 5)],
        3: [(5, 5), (5, 7), (7, 5)],
        4: [(7, 7)],
    }
    for _ in range(300):
        h, w = random.choice(dim_choices[n])
        mask = np.ones((h, w), dtype=bool)
        full_area = h * w
        min_removed = max(1, full_area - 40)
        max_removed = full_area - 20
        if n == 4:
            min_removed = max(min_removed, 9)
        goal = random.randint(min_removed, max_removed)
        templates = _bite_templates(h, w, n)
        random.shuffle(templates)
        removed: set[tuple[int, int]] = set()
        for tpl in templates:
            new_removed = removed | tpl
            if len(new_removed) > max_removed:
                continue
            if len(removed) < min_removed or random.random() < 0.6:
                removed = new_removed
            if len(removed) >= goal and random.random() < 0.5:
                break
        if not (min_removed <= len(removed) <= max_removed):
            continue
        for r, c in removed:
            mask[r, c] = False
        if not _is_connected_true(mask):
            continue
        coords = np.argwhere(mask)
        if _bbox(coords) != (0, 0, h - 1, w - 1):
            continue
        pattern = _pattern_mask_local((h, w), n)
        if not np.any(pattern & ~mask):
            continue
        layer_ok = True
        cr, cc = h // 2, w // 2
        for dist in range(n):
            visible = False
            for r, c in coords.tolist():
                if max(abs(int(r) - cr), abs(int(c) - cc)) == dist:
                    visible = True
                    break
            if not visible:
                layer_ok = False
                break
        if not layer_ok:
            continue
        outer_crop = False
        for r in range(h):
            for c in range(w):
                if max(abs(r - cr), abs(c - cc)) == n - 1 and pattern[r, c] and not mask[r, c]:
                    outer_crop = True
                    break
            if outer_crop:
                break
        if not outer_crop:
            continue
        return mask
    raise RuntimeError("failed to sample a cavity mask")


def _border_from_cavity(cavity: np.ndarray) -> np.ndarray:
    h, w = cavity.shape
    shell = np.zeros((h + 2, w + 2), dtype=bool)
    for r, c in np.argwhere(cavity):
        rr, cc = int(r) + 1, int(c) + 1
        for dr, dc in NEIGHBORS8:
            shell[rr + dr, cc + dc] = True
    shell[1 : h + 1, 1 : w + 1] &= ~cavity
    return shell


def _place_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _can_place_cells(grid: np.ndarray, cells: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if not _in_bounds(grid.shape, r, c) or grid[r, c] != BLACK:
            return False
    for r, c in cells:
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if _in_bounds(grid.shape, nr, nc) and (nr, nc) not in cells and grid[nr, nc] != BLACK:
                return False
    return True


def _attempt_place_blob(grid: np.ndarray, blob: set[tuple[int, int]], color: int) -> set[tuple[int, int]] | None:
    h = max(r for r, _ in blob) + 1
    w = max(c for _, c in blob) + 1
    candidates = [
        (r0, c0)
        for r0 in range(grid.shape[0] - h + 1)
        for c0 in range(grid.shape[1] - w + 1)
    ]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        placed = {(r0 + r, c0 + c) for r, c in blob}
        if _can_place_cells(grid, placed):
            _place_cells(grid, placed, color)
            return placed
    return None


def _truth_output(grid: np.ndarray, cavity_abs: set[tuple[int, int]], objects: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    ordered = sorted(objects, key=lambda obj: int(obj["area"]), reverse=True)
    colors = [int(obj["color"]) for obj in ordered]
    cavity = np.array(sorted(cavity_abs), dtype=int)
    return _paint_cavity_rings(out, cavity, colors)


def _semantic_checks(grid: np.ndarray, truth: dict) -> None:
    scene = _extract_scene(grid)
    assert _component_set(scene["cavity"]) == truth["cavity"]
    assert scene["border"]["color"] == truth["border_color"]
    assert _component_set(scene["border"]["coords"]) == truth["border"]

    seen_objects = sorted(
        [(int(obj["color"]), int(obj["area"])) for obj in scene["objects"]],
        key=lambda x: (x[1], x[0]),
    )
    truth_objects = sorted(
        [(int(obj["color"]), int(obj["area"])) for obj in truth["objects"]],
        key=lambda x: (x[1], x[0]),
    )
    assert seen_objects == truth_objects

    ordered = sorted(truth["objects"], key=lambda obj: int(obj["area"]), reverse=True)
    ordered_colors = [int(obj["color"]) for obj in ordered]
    assert ordered_colors != sorted(ordered_colors)
    assert ordered_colors != sorted(ordered_colors, reverse=True)

    tags_by_area = [int(obj["tag"]) for obj in ordered]
    tags_by_row = [int(obj["tag"]) for obj in sorted(truth["objects"], key=lambda obj: obj["centroid"])]
    tags_by_col = [
        int(obj["tag"]) for obj in sorted(truth["objects"], key=lambda obj: (obj["centroid"][1], obj["centroid"][0]))
    ]
    assert tags_by_area != tags_by_row or tags_by_area != tags_by_col

    expected = _truth_output(grid, truth["cavity"], truth["objects"])
    actual = main(grid)
    assert np.array_equal(actual, expected)

    cavity = np.array(sorted(truth["cavity"]), dtype=int)
    bbox = _bbox(cavity)
    cr, cc = _center_from_bbox(bbox)
    n = len(truth["objects"])
    for dist in range(n):
        assert any(max(abs(int(r) - cr), abs(int(c) - cc)) == dist for r, c in truth["cavity"])
    assert any(
        max(abs(r - cr), abs(c - cc)) == n - 1 and (r, c) not in truth["cavity"]
        for r in range(bbox[0], bbox[2] + 1)
        for c in range(bbox[1], bbox[3] + 1)
        if max(abs(r - cr), abs(c - cc)) < n
    )


def _generate_instance() -> tuple[np.ndarray, dict]:
    n_objects = random.choices([2, 3, 4], weights=[2, 4, 3], k=1)[0]
    cavity_local = _sample_cavity_mask(n_objects)
    border_local = _border_from_cavity(cavity_local)
    sprite_h, sprite_w = border_local.shape

    grid_h = random.randint(max(15, sprite_h + 6), 18)
    grid_w = random.randint(max(15, sprite_w + 6), 18)
    grid = np.zeros((grid_h, grid_w), dtype=int)

    colors = random.sample(PALETTE, n_objects + 1)
    border_color = int(colors[0])
    object_colors = [int(c) for c in colors[1:]]
    areas = random.sample(range(2, 11), n_objects)

    sr0 = random.randint(2, grid_h - sprite_h - 2)
    sc0 = random.randint(2, grid_w - sprite_w - 2)

    border_abs: set[tuple[int, int]] = set()
    for r, c in np.argwhere(border_local):
        border_abs.add((sr0 + int(r), sc0 + int(c)))
    _place_cells(grid, border_abs, border_color)

    cavity_abs: set[tuple[int, int]] = set()
    for r, c in np.argwhere(cavity_local):
        cavity_abs.add((sr0 + 1 + int(r), sc0 + 1 + int(c)))

    blob_specs = list(zip(areas, object_colors))
    random.shuffle(blob_specs)
    objects: list[dict] = []
    for tag, (area, color) in enumerate(blob_specs):
        blob = _random_blob_cells(int(area))
        placed = _attempt_place_blob(grid, blob, int(color))
        if placed is None:
            raise RuntimeError("failed to place an object")
        coords = np.array(sorted(placed), dtype=int)
        centroid = (float(coords[:, 0].mean()), float(coords[:, 1].mean()))
        objects.append(
            {
                "tag": int(tag),
                "color": int(color),
                "area": int(area),
                "coords": placed,
                "centroid": centroid,
            }
        )

    truth = {
        "border_color": int(border_color),
        "border": border_abs,
        "cavity": cavity_abs,
        "objects": objects,
    }
    return grid, truth


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        try:
            grid, truth = _generate_instance()
            _semantic_checks(grid, truth)
            return grid
        except (AssertionError, RuntimeError, ValueError):
            continue
    raise RuntimeError("failed to generate a valid input")
