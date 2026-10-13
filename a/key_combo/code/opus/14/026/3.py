from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
TEMPLATE_COLOR = 1
OBJECT_COLOR = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _count_holes(mask: np.ndarray) -> int:
    """Count 4-connected enclosed background regions inside a foreground mask."""
    padded_bg = np.pad(~mask.astype(bool), 1, constant_values=True)
    h, w = padded_bg.shape
    seen = np.zeros((h, w), dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if padded_bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if padded_bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and padded_bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    holes = 0
    for r in range(h):
        for c in range(w):
            if not padded_bg[r, c] or seen[r, c]:
                continue
            holes += 1
            seen[r, c] = True
            q.append((r, c))
            while q:
                rr, cc = q.popleft()
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and padded_bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return holes


def _normalize(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, _, _ = _bbox(cells)
    return sorted((r - r0, c - c0) for r, c in cells)


def _transform_template(cells: list[tuple[int, int]], op: int) -> list[tuple[int, int]]:
    arr = np.zeros((3, 3), dtype=int)
    for r, c in cells:
        arr[r, c] = 1
    if op == 1:
        arr = np.fliplr(arr)
    elif op == 2:
        arr = np.flipud(arr)
    elif op == 3:
        arr = np.flipud(np.fliplr(arr))
    return _normalize([(int(r), int(c)) for r, c in np.argwhere(arr == 1)])


def _stamp_scaled(
    out: np.ndarray,
    rel_cells: list[tuple[int, int]],
    origin_r: int,
    origin_c: int,
    scale: int,
    color: int,
) -> None:
    h, w = out.shape
    for rr, cc in rel_cells:
        top = origin_r + rr * scale
        left = origin_c + cc * scale
        for dr in range(scale):
            for dc in range(scale):
                r = top + dr
                c = left + dc
                if 0 <= r < h and 0 <= c < w:
                    out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count holes in each gray object and stamp a scaled blue template at that object's position.

    Input structure:
    - A black background contains one small blue template component confined to the upper-left 3x3 area.
    - Elsewhere there are 2-3 gray 4-connected foreground objects; each is a solid rectangular block with 1-3 enclosed black holes.
    - The blue template is asymmetric and its bbox is 2x3 or 3x3.

    Rule:
    - Use the blue component in the upper-left corner as the template coordinate system.
    - For each gray object, count its enclosed 4-connected background holes to get a scale k in {1,2,3}.
    - Erase the gray object, then stamp the template scaled by factor k with top-left alignment to the gray object's bbox; clip any off-grid part.
    - Remove the original upper-left template from the output.

    Key insight:
    - The gray objects do not contribute their own shape to the output; only their hole counts matter, because hole count determines the template scale.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    blue_components = _find_components_of_color(g, TEMPLATE_COLOR)
    template_candidates = [comp for comp in blue_components if _bbox(comp)[2] <= 2 and _bbox(comp)[3] <= 2]
    assert len(template_candidates) == 1, "expected exactly one upper-left blue template"
    template = template_candidates[0]
    tr0, tc0, _, _ = _bbox(template)
    template_rel = sorted((r - tr0, c - tc0) for r, c in template)

    out = np.zeros_like(g)
    for comp in _find_components_of_color(g, OBJECT_COLOR):
        r0, c0, r1, c1 = _bbox(comp)
        local = g[r0 : r1 + 1, c0 : c1 + 1] == int(OBJECT_COLOR)
        scale = _count_holes(local)
        assert scale in (1, 2, 3), "each object must encode 1-3 holes"
        _stamp_scaled(out, template_rel, r0, c0, scale, TEMPLATE_COLOR)

    return out


TEMPLATE_LIBRARY = [
    [(0, 0), (0, 1), (1, 1), (1, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 0)],
    [(0, 0), (1, 0), (1, 1), (1, 2)],
    [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)],
    [(0, 1), (1, 0), (1, 1), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
]

HOLE_SHAPES_BY_AREA = {
    1: [(1, 1)],
    2: [(1, 2), (2, 1)],
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
}

AREA_PARTITIONS = {
    3: {
        1: [[3]],
        2: [[1, 2]],
        3: [[1, 1, 1]],
    },
    4: {
        1: [[4]],
        2: [[1, 3], [2, 2]],
        3: [[1, 1, 2]],
    },
}


def _template_has_required_bbox(rel_cells: list[tuple[int, int]]) -> bool:
    rows = [r for r, _ in rel_cells]
    cols = [c for _, c in rel_cells]
    h = max(rows) - min(rows) + 1
    w = max(cols) - min(cols) + 1
    return (h, w) in {(2, 3), (3, 3)}


def _choose_template() -> list[tuple[int, int]]:
    base = random.choice(TEMPLATE_LIBRARY)
    rel = _transform_template(base, random.randrange(4))
    if not _template_has_required_bbox(rel):
        rel = _normalize(base)
    return rel


def _rectangles_too_close(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    pad: int,
) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _partition_areas(count: int, total_area: int) -> list[int]:
    return random.choice(AREA_PARTITIONS[total_area][count])[:]


def _choose_shape(area: int) -> tuple[int, int]:
    return random.choice(HOLE_SHAPES_BY_AREA[area])


def _hole_cells(r0: int, c0: int, h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]


def _holes_separated(existing: list[list[tuple[int, int]]], candidate: list[tuple[int, int]]) -> bool:
    occupied = {cell for hole in existing for cell in hole}
    for r, c in candidate:
        if (r, c) in occupied:
            return False
        for rr, cc in occupied:
            if abs(r - rr) + abs(c - cc) < 2:
                return False
    return True


def _make_object_with_holes(h: int, w: int, hole_count: int, total_hole_area: int) -> np.ndarray | None:
    area_plan = _partition_areas(hole_count, total_hole_area)
    random.shuffle(area_plan)
    for _ in range(80):
        holes: list[list[tuple[int, int]]] = []
        ok = True
        for area in area_plan:
            shape_h, shape_w = _choose_shape(area)
            candidates = []
            for r0 in range(1, h - shape_h):
                for c0 in range(1, w - shape_w):
                    cells = _hole_cells(r0, c0, shape_h, shape_w)
                    if _holes_separated(holes, cells):
                        candidates.append(cells)
            if not candidates:
                ok = False
                break
            holes.append(random.choice(candidates))
        if not ok:
            continue
        obj = np.full((h, w), OBJECT_COLOR, dtype=int)
        for hole in holes:
            for r, c in hole:
                obj[r, c] = BG
        if _count_holes(obj == OBJECT_COLOR) == hole_count:
            return obj
    return None


def _build_object_specs(counts: list[int]) -> list[dict] | None:
    outer_h, outer_w = random.choice([(6, 6), (6, 7), (7, 6), (7, 7)])
    total_hole_area = random.choice([3, 4])
    specs: list[dict] = []
    for hole_count in counts:
        obj = _make_object_with_holes(outer_h, outer_w, hole_count, total_hole_area)
        if obj is None:
            return None
        specs.append(
            {
                "hole_count": hole_count,
                "patch": obj,
                "bbox_h": outer_h,
                "bbox_w": outer_w,
                "fg_area": int(np.count_nonzero(obj == OBJECT_COLOR)),
            }
        )

    fg_areas = {spec["fg_area"] for spec in specs}
    if len(fg_areas) != 1:
        return None
    return specs


def _choose_counts() -> list[int]:
    choice = random.random()
    if choice < 0.5:
        return [1, 2, 3]
    if choice < 0.7:
        return [1, 2]
    if choice < 0.85:
        return [1, 3]
    return [2, 3]


def _stamp_bbox(
    origin_r: int,
    origin_c: int,
    template_rel: list[tuple[int, int]],
    scale: int,
    grid_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    rows = [origin_r + r * scale for r, _ in template_rel]
    cols = [origin_c + c * scale for _, c in template_rel]
    h = scale
    w = scale
    r0 = min(rows)
    c0 = min(cols)
    r1 = max(r + h - 1 for r in rows)
    c1 = max(c + w - 1 for c in cols)
    gh, gw = grid_shape
    return max(0, r0), max(0, c0), min(gh - 1, r1), min(gw - 1, c1)


def _place_objects(
    grid_h: int,
    grid_w: int,
    template_rel: list[tuple[int, int]],
    specs: list[dict],
    force_clip: bool,
) -> bool:
    obj_boxes: list[tuple[int, int, int, int]] = []
    stamp_boxes: list[tuple[int, int, int, int]] = []
    template_box = (0, 0, 2, 2)
    order = sorted(range(len(specs)), key=lambda i: specs[i]["hole_count"], reverse=True)

    clip_index = order[0] if force_clip else None

    for idx in order:
        spec = specs[idx]
        h = spec["bbox_h"]
        w = spec["bbox_w"]
        scale = spec["hole_count"]
        stamp_h = (max(r for r, _ in template_rel) + 1) * scale
        stamp_w = (max(c for _, c in template_rel) + 1) * scale
        candidates = []
        for r0 in range(0, grid_h - h + 1):
            for c0 in range(0, grid_w - w + 1):
                if r0 <= 2 and c0 <= 2:
                    continue
                obj_box = (r0, c0, r0 + h - 1, c0 + w - 1)
                if _rectangles_too_close(obj_box, template_box, 1):
                    continue
                if any(_rectangles_too_close(obj_box, other, 1) for other in obj_boxes):
                    continue
                clipped = r0 + stamp_h > grid_h or c0 + stamp_w > grid_w
                if idx == clip_index and not clipped:
                    continue
                if idx != clip_index and clipped and random.random() < 0.6:
                    continue
                stamp_box = _stamp_bbox(r0, c0, template_rel, scale, (grid_h, grid_w))
                if any(_rectangles_too_close(stamp_box, other, 1) for other in stamp_boxes):
                    continue
                candidates.append((r0, c0, stamp_box, clipped))
        if not candidates:
            return False
        r0, c0, stamp_box, clipped = random.choice(candidates)
        spec["r0"] = r0
        spec["c0"] = c0
        spec["clipped"] = clipped
        obj_boxes.append((r0, c0, r0 + h - 1, c0 + w - 1))
        stamp_boxes.append(stamp_box)

    return True


def _render_input(grid_h: int, grid_w: int, template_rel: list[tuple[int, int]], specs: list[dict]) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for r, c in template_rel:
        grid[r, c] = TEMPLATE_COLOR
    for spec in specs:
        r0 = spec["r0"]
        c0 = spec["c0"]
        patch = spec["patch"]
        h, w = patch.shape
        grid[r0 : r0 + h, c0 : c0 + w] = patch
    return grid


def _render_expected(grid_h: int, grid_w: int, template_rel: list[tuple[int, int]], specs: list[dict]) -> np.ndarray:
    out = np.zeros((grid_h, grid_w), dtype=int)
    for spec in specs:
        _stamp_scaled(out, template_rel, spec["r0"], spec["c0"], spec["hole_count"], TEMPLATE_COLOR)
    return out


def _validate_sample(inp: np.ndarray, expected: np.ndarray, specs: list[dict], template_rel: list[tuple[int, int]], force_clip: bool) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    blue_comps = _find_components_of_color(inp, TEMPLATE_COLOR)
    if len(blue_comps) != 1:
        return False
    if _normalize(blue_comps[0]) != _normalize([(r, c) for r, c in template_rel]):
        return False

    gray_comps = _find_components_of_color(inp, OBJECT_COLOR)
    if len(gray_comps) != len(specs):
        return False
    found = []
    for comp in gray_comps:
        r0, c0, r1, c1 = _bbox(comp)
        local = inp[r0 : r1 + 1, c0 : c1 + 1] == OBJECT_COLOR
        holes = _count_holes(local)
        fg_area = int(np.count_nonzero(local))
        found.append((r0, c0, holes, fg_area))
    found.sort()
    intended = sorted((spec["r0"], spec["c0"], spec["hole_count"], spec["fg_area"]) for spec in specs)
    if found != intended:
        return False
    if len({spec["hole_count"] for spec in specs}) != len(specs):
        return False
    if len({spec["fg_area"] for spec in specs}) != 1:
        return False
    if force_clip and not any(bool(spec["clipped"]) for spec in specs):
        return False
    if np.any(expected[:3, :3] != BG):
        return False

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False
    return True


def _generate_once() -> np.ndarray | None:
    grid_h = random.randint(18, 24)
    grid_w = random.randint(18, 24)
    template_rel = _choose_template()
    counts = _choose_counts()
    specs = _build_object_specs(counts)
    if specs is None:
        return None

    force_clip = random.random() < 0.45
    if not _place_objects(grid_h, grid_w, template_rel, specs, force_clip):
        return None

    inp = _render_input(grid_h, grid_w, template_rel, specs)
    expected = _render_expected(grid_h, grid_w, template_rel, specs)
    if not _validate_sample(inp, expected, specs, template_rel, force_clip):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _generate_once()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid hole-count scaling sample")
