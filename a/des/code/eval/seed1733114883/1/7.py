from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
COLOR_POOL = tuple(range(1, 10))
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_MIN = 18
GRID_MAX = 32
MAX_GENERATION_ATTEMPTS = 400
CLUSTER_BUILD_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 160
BORDER_MARGIN = 1
CLUSTER_CLEARANCE = 2


@dataclass
class _Object:
    bbox: tuple[int, int, int, int]  # (r0, c0, r1, c1), exclusive
    pattern: np.ndarray
    base_color: int

    @property
    def area(self) -> int:
        r0, c0, r1, c1 = self.bbox
        return int((r1 - r0) * (c1 - c0))


@dataclass
class _Rect:
    r: int
    c: int
    h: int
    w: int
    color: int = -1

    @property
    def area(self) -> int:
        return int(self.h * self.w)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return int(self.r), int(self.c), int(self.r + self.h), int(self.c + self.w)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _parse_initial_objects(grid: np.ndarray) -> list[_Object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects: list[_Object] = []
    seen_colors: set[int] = set()
    for coords in _components(g != BLACK):
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0 = int(rows.min())
        c0 = int(cols.min())
        r1 = int(rows.max()) + 1
        c1 = int(cols.max()) + 1

        colors = np.unique(g[rows, cols])
        assert colors.size == 1, "each input object must be monochromatic"
        color = int(colors[0])
        assert color != BLACK, "objects must be non-black"
        assert color not in seen_colors, "input object colors must be unique"
        patch = g[r0:r1, c0:c1]
        assert bool(np.all(patch == color)), "each input object must be a solid rectangle"

        seen_colors.add(color)
        objects.append(
            _Object(
                bbox=(r0, c0, r1, c1),
                pattern=np.full((r1 - r0, c1 - c0), color, dtype=int),
                base_color=color,
            )
        )

    assert objects, "expected at least one object"
    areas = [obj.area for obj in objects]
    assert len(areas) == len(set(areas)), "input object areas must be unique"
    return objects


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(int(a1), int(b1)) - max(int(a0), int(b0)))


def _adjacent(a: _Object, b: _Object) -> bool:
    ar0, ac0, ar1, ac1 = a.bbox
    br0, bc0, br1, bc1 = b.bbox

    if (ac1 + 1 == bc0 or bc1 + 1 == ac0) and _interval_overlap(ar0, ar1, br0, br1) > 0:
        return True
    if (ar1 + 1 == br0 or br1 + 1 == ar0) and _interval_overlap(ac0, ac1, bc0, bc1) > 0:
        return True
    return False


def _merge_objects(big: _Object, small: _Object) -> _Object:
    br0, bc0, br1, bc1 = big.bbox
    sr0, sc0, sr1, sc1 = small.bbox

    r0 = min(br0, sr0)
    c0 = min(bc0, sc0)
    r1 = max(br1, sr1)
    c1 = max(bc1, sc1)

    merged = np.full((r1 - r0, c1 - c0), int(big.base_color), dtype=int)
    merged[br0 - r0 : br1 - r0, bc0 - c0 : bc1 - c0] = big.pattern
    merged[sr0 - r0 : sr1 - r0, sc0 - c0 : sc1 - c0] = small.pattern
    return _Object(bbox=(r0, c0, r1, c1), pattern=merged, base_color=int(big.base_color))


def _pick_next_merge(objects: list[_Object]) -> tuple[int, int] | None:
    candidates: list[tuple[int, tuple[int, int, int, int], int, tuple[int, int, int, int], int, int]] = []
    for i, big in enumerate(objects):
        smaller: list[tuple[int, tuple[int, int, int, int], int]] = []
        for j, small in enumerate(objects):
            if i == j or big.area <= small.area or not _adjacent(big, small):
                continue
            smaller.append((int(small.area), tuple(map(int, small.bbox)), int(j)))
        if not smaller:
            continue
        _area, _bbox, prey_idx = min(smaller, key=lambda item: (item[0], item[1], item[2]))
        prey = objects[prey_idx]
        candidates.append(
            (
                int(prey.area),
                tuple(map(int, big.bbox)),
                int(big.area),
                tuple(map(int, prey.bbox)),
                int(i),
                int(prey_idx),
            )
        )

    if not candidates:
        return None
    _prey_area, _pred_bbox, _pred_area, _prey_bbox, i, j = min(candidates)
    return int(i), int(j)


def _run_process(objects: list[_Object]) -> tuple[list[_Object], list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]]:
    current = list(objects)
    history: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    while True:
        pair = _pick_next_merge(current)
        if pair is None:
            return current, history
        i, j = pair
        big = current[i]
        small = current[j]
        history.append((tuple(map(int, big.bbox)), tuple(map(int, small.bbox))))
        merged = _merge_objects(big, small)
        survivors = [obj for idx, obj in enumerate(current) if idx not in (i, j)]
        survivors.append(merged)
        current = survivors


def _render(shape: tuple[int, int], objects: list[_Object]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for obj in objects:
        r0, c0, r1, c1 = obj.bbox
        assert bool(np.all(out[r0:r1, c0:c1] == BLACK)), "final objects must remain disjoint"
        out[r0:r1, c0:c1] = obj.pattern
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeatedly let each larger rectangle swallow its adjacent smaller neighbors.

    Input structure
    - Background is black.
    - Each non-black object is a solid monochromatic rectangle.
    - Object colors are unique and object areas are unique.
    - Adjacent objects are separated by exactly one black cell horizontally or
      vertically, with overlapping projection on the other axis.

    Rule
    - Repeatedly choose a swallow operation until no adjacent larger/smaller pair remains.
    - For each large object, only its smallest adjacent smaller neighbor is eligible.
    - Swallowing replaces the large object by the union bounding rectangle of
      the large rectangle, the 1-cell gap, and the small rectangle.
    - New cells are filled with the large object's outer color, while the entire
      small object's current pattern is preserved as an internal trace.
    """
    g = np.asarray(grid, dtype=int)
    objects = _parse_initial_objects(g)
    final_objects, _history = _run_process(objects)
    return _render(g.shape, final_objects)


def _randint(lo: int, hi: int) -> int:
    return random.randint(int(lo), int(hi))


def _partition_total(total: int, parts: int, min_each: int) -> list[int] | None:
    if parts <= 0 or total < parts * min_each:
        return None
    remaining = int(total)
    out: list[int] = []
    for idx in range(parts):
        slots_left = parts - idx - 1
        lo = int(min_each)
        hi = int(remaining - slots_left * min_each)
        out.append(_randint(lo, hi))
        remaining -= out[-1]
    random.shuffle(out)
    return out


def _pick_dims(
    used_areas: set[int],
    h_range: tuple[int, int],
    w_range: tuple[int, int],
    *,
    min_area: int | None = None,
    max_area: int | None = None,
) -> tuple[int, int] | None:
    options = []
    for h in range(int(h_range[0]), int(h_range[1]) + 1):
        for w in range(int(w_range[0]), int(w_range[1]) + 1):
            area = int(h * w)
            if area in used_areas:
                continue
            if min_area is not None and area < int(min_area):
                continue
            if max_area is not None and area > int(max_area):
                continue
            options.append((int(h), int(w)))
    return random.choice(options) if options else None


def _normalize_rects(rects: list[_Rect]) -> list[_Rect]:
    min_r = min(rect.r for rect in rects)
    min_c = min(rect.c for rect in rects)
    return [_Rect(rect.r - min_r, rect.c - min_c, rect.h, rect.w, rect.color) for rect in rects]


def _cluster_shape(rects: list[_Rect]) -> tuple[int, int]:
    max_r = max(rect.r + rect.h for rect in rects)
    max_c = max(rect.c + rect.w for rect in rects)
    return int(max_r), int(max_c)


def _build_line_cluster(used_areas: set[int]) -> list[_Rect] | None:
    direction = random.choice(("right", "left", "down", "up"))
    n_leaves = random.choice((2, 3))
    horizontal = direction in {"right", "left"}

    for _ in range(CLUSTER_BUILD_ATTEMPTS):
        local_used = set(used_areas)

        if horizontal:
            root = _pick_dims(local_used, (max(4, 2 * n_leaves), 8), (3, 6))
            if root is None:
                continue
            root_h, root_w = root
            root_area = root_h * root_w
            local_used.add(root_area)
            lane_sizes = _partition_total(root_h, n_leaves, 2)
            if lane_sizes is None:
                continue

            rects = [_Rect(0, 0, root_h, root_w)]
            lane_start = 0
            cursor = root_w + 1
            ok = True
            for lane_h in lane_sizes:
                dims = _pick_dims(local_used, (2, lane_h), (2, 4), max_area=root_area - 1)
                if dims is None:
                    ok = False
                    break
                leaf_h, leaf_w = dims
                row = lane_start + _randint(0, lane_h - leaf_h)
                col = cursor if direction == "right" else -(cursor + leaf_w)
                rects.append(_Rect(row, col, leaf_h, leaf_w))
                local_used.add(leaf_h * leaf_w)
                lane_start += lane_h
                cursor += leaf_w + 1
            if not ok:
                continue
        else:
            root = _pick_dims(local_used, (3, 6), (max(4, 2 * n_leaves), 8))
            if root is None:
                continue
            root_h, root_w = root
            root_area = root_h * root_w
            local_used.add(root_area)
            lane_sizes = _partition_total(root_w, n_leaves, 2)
            if lane_sizes is None:
                continue

            rects = [_Rect(0, 0, root_h, root_w)]
            lane_start = 0
            cursor = root_h + 1
            ok = True
            for lane_w in lane_sizes:
                dims = _pick_dims(local_used, (2, 4), (2, lane_w), max_area=root_area - 1)
                if dims is None:
                    ok = False
                    break
                leaf_h, leaf_w = dims
                row = cursor if direction == "down" else -(cursor + leaf_h)
                col = lane_start + _randint(0, lane_w - leaf_w)
                rects.append(_Rect(row, col, leaf_h, leaf_w))
                local_used.add(leaf_h * leaf_w)
                lane_start += lane_w
                cursor += leaf_h + 1
            if not ok:
                continue

        used_areas.update(rect.area for rect in rects)
        return _normalize_rects(rects)

    return None


def _build_star_cluster(used_areas: set[int]) -> list[_Rect] | None:
    sides = random.sample(("left", "right", "up", "down"), k=random.choice((2, 3)))

    for _ in range(CLUSTER_BUILD_ATTEMPTS):
        local_used = set(used_areas)
        root = _pick_dims(local_used, (4, 7), (4, 7))
        if root is None:
            continue
        root_h, root_w = root
        root_area = root_h * root_w
        local_used.add(root_area)

        rects = [_Rect(0, 0, root_h, root_w)]
        ok = True
        for side in sides:
            if side in {"left", "right"}:
                dims = _pick_dims(local_used, (2, root_h), (2, 4), max_area=root_area - 1)
                if dims is None:
                    ok = False
                    break
                leaf_h, leaf_w = dims
                row = _randint(0, root_h - leaf_h)
                col = -(leaf_w + 1) if side == "left" else root_w + 1
            else:
                dims = _pick_dims(local_used, (2, 4), (2, root_w), max_area=root_area - 1)
                if dims is None:
                    ok = False
                    break
                leaf_h, leaf_w = dims
                row = -(leaf_h + 1) if side == "up" else root_h + 1
                col = _randint(0, root_w - leaf_w)

            rects.append(_Rect(row, col, leaf_h, leaf_w))
            local_used.add(leaf_h * leaf_w)
        if not ok:
            continue

        used_areas.update(rect.area for rect in rects)
        return _normalize_rects(rects)

    return None


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + gap <= br0
        or br1 + gap <= ar0
        or ac1 + gap <= bc0
        or bc1 + gap <= ac0
    )


def _place_clusters(clusters: list[list[_Rect]]) -> np.ndarray | None:
    shapes = [_cluster_shape(cluster) for cluster in clusters]
    n_clusters = len(clusters)
    max_h = max(shape[0] for shape in shapes)
    max_w = max(shape[1] for shape in shapes)
    sum_h = sum(shape[0] for shape in shapes)
    sum_w = sum(shape[1] for shape in shapes)

    vertical_stack = random.choice((True, False))
    if vertical_stack:
        min_h = max(GRID_MIN, sum_h + 3 * (n_clusters - 1) + 2 * BORDER_MARGIN)
        min_w = max(GRID_MIN, max_w + 2 * BORDER_MARGIN)
    else:
        min_h = max(GRID_MIN, max_h + 2 * BORDER_MARGIN)
        min_w = max(GRID_MIN, sum_w + 3 * (n_clusters - 1) + 2 * BORDER_MARGIN)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    h = _randint(min_h, GRID_MAX)
    w = _randint(min_w, GRID_MAX)
    grid = np.full((h, w), BLACK, dtype=int)

    placed_boxes: list[tuple[int, int, int, int]] = []
    origins: list[tuple[int, int]] = []
    order = list(range(n_clusters))
    random.shuffle(order)

    for idx in order:
        ch, cw = shapes[idx]
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = _randint(BORDER_MARGIN, h - ch - BORDER_MARGIN)
            c0 = _randint(BORDER_MARGIN, w - cw - BORDER_MARGIN)
            bbox = (int(r0), int(c0), int(r0 + ch), int(c0 + cw))
            if all(_separated(bbox, other, CLUSTER_CLEARANCE) for other in placed_boxes):
                placed_boxes.append(bbox)
                origins.append((idx, r0, c0))
                placed = True
                break
        if not placed:
            return None

    origin_map = {idx: (r0, c0) for idx, r0, c0 in origins}
    for idx, cluster in enumerate(clusters):
        r_shift, c_shift = origin_map[idx]
        for rect in cluster:
            rr = r_shift + rect.r
            cc = c_shift + rect.c
            assert rect.color != -1, "all rectangles must have assigned colors"
            if not bool(np.all(grid[rr : rr + rect.h, cc : cc + rect.w] == BLACK)):
                return None
            grid[rr : rr + rect.h, cc : cc + rect.w] = int(rect.color)

    return grid


def _build_candidate() -> np.ndarray | None:
    used_areas: set[int] = set()
    clusters: list[list[_Rect]] = []

    first_builder = random.choice((_build_line_cluster, _build_star_cluster))
    first = first_builder(used_areas)
    if first is None:
        return None
    clusters.append(first)

    if random.random() < 0.55:
        second_builder = random.choice((_build_line_cluster, _build_star_cluster))
        second = second_builder(used_areas)
        if second is not None:
            clusters.append(second)

    rects = [rect for cluster in clusters for rect in cluster]
    if len(rects) > len(COLOR_POOL):
        return None

    colors = random.sample(COLOR_POOL, k=len(rects))
    for rect, color in zip(rects, colors, strict=False):
        rect.color = int(color)

    grid = _place_clusters(clusters)
    if grid is None:
        return None

    try:
        objects = _parse_initial_objects(grid)
        final_objects, history = _run_process(objects)
        if not history:
            return None
        out = _render(grid.shape, final_objects)
        if bool(np.array_equal(grid, out)):
            return None
        if not bool(np.array_equal(main(grid), out)):
            return None
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 7")
