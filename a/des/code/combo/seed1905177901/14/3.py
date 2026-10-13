from __future__ import annotations

import numpy as np


BG = 0
PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
COUNT_OPTIONS = np.array([4, 6, 8, 9, 10, 12], dtype=int)
OBJECT_COUNT_OPTIONS = np.array([2, 3], dtype=int)
SCATTER_RADIUS = 4
OVERLAP_PROB = 0.4
MAX_GEN_ATTEMPTS = 400

_RNG = np.random.default_rng()


def _colors_in_scan_order(grid: np.ndarray) -> list[int]:
    seen: set[int] = set()
    colors: list[int] = []
    for value in grid.reshape(-1):
        color = int(value)
        if color != BG and color not in seen:
            seen.add(color)
            colors.append(color)
    return colors


def _best_factor_pair(n: int) -> tuple[int, int]:
    best_h = 1
    for h in range(1, int(np.sqrt(n)) + 1):
        if n % h == 0:
            best_h = h
    return best_h, n // best_h


def _centroid_floor(coords: np.ndarray) -> tuple[int, int]:
    count = int(coords.shape[0])
    return int(coords[:, 0].sum() // count), int(coords[:, 1].sum() // count)


def _rect_from_center(center_r: int, center_c: int, rect_h: int, rect_w: int) -> tuple[int, int, int, int]:
    # This anchor makes floor(pixel-centroid) equal the requested center even for even sizes.
    top = center_r - (rect_h - 1) // 2
    left = center_c - (rect_w - 1) // 2
    return top, left, top + rect_h - 1, left + rect_w - 1


def _bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _bbox_intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    if not _bbox_intersects(a, b):
        return 0
    rows = min(a[2], b[2]) - max(a[0], b[0]) + 1
    cols = min(a[3], b[3]) - max(a[1], b[1]) + 1
    return int(rows * cols)


def _expand_bbox(bbox: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    return bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad


def _render_groups(shape: tuple[int, int], groups: list[dict[str, int]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for group in sorted(groups, key=lambda item: (item["count"], item["order"])):
        top, left, bottom, right = group["bbox"]
        out[top : bottom + 1, left : right + 1] = group["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gather each color's scattered pixels into a compact rectangle whose area is the
    color count and whose placement is determined by the floor centroid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    groups: list[dict[str, int]] = []
    for order, color in enumerate(_colors_in_scan_order(g)):
        coords = np.argwhere(g == color)
        count = int(coords.shape[0])
        rect_h, rect_w = _best_factor_pair(count)
        center_r, center_c = _centroid_floor(coords)
        bbox = _rect_from_center(center_r, center_c, rect_h, rect_w)
        top, left, bottom, right = bbox
        assert 0 <= top <= bottom < height
        assert 0 <= left <= right < width
        groups.append(
            {
                "order": order,
                "color": int(color),
                "count": count,
                "bbox": bbox,
            }
        )

    return _render_groups((height, width), groups)


def _cells_far_enough(cell: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
    return all(max(abs(cell[0] - r), abs(cell[1] - c)) > 1 for r, c in chosen)


def _scatter_points(
    height: int,
    width: int,
    center: tuple[int, int],
    count: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    center_r, center_c = center
    need_pairs = count // 2

    for _ in range(80):
        chosen: list[tuple[int, int]] = []
        chosen_set: set[tuple[int, int]] = set()

        if count % 2 == 1:
            center_cell = (center_r, center_c)
            if center_cell in occupied:
                return None
            chosen.append(center_cell)
            chosen_set.add(center_cell)

        offsets = [
            (dr, dc)
            for dr in range(-SCATTER_RADIUS, SCATTER_RADIUS + 1)
            for dc in range(-SCATTER_RADIUS, SCATTER_RADIUS + 1)
            if dr > 0 or (dr == 0 and dc > 0)
        ]
        _RNG.shuffle(offsets)

        pair_count = 0
        for dr, dc in offsets:
            p0 = (center_r + dr, center_c + dc)
            p1 = (center_r - dr, center_c - dc)
            if p0 == p1:
                continue
            if not (0 <= p0[0] < height and 0 <= p0[1] < width):
                continue
            if not (0 <= p1[0] < height and 0 <= p1[1] < width):
                continue
            if p0 in occupied or p1 in occupied or p0 in chosen_set or p1 in chosen_set:
                continue
            if not _cells_far_enough(p0, chosen) or not _cells_far_enough(p1, chosen):
                continue
            chosen.extend([p0, p1])
            chosen_set.add(p0)
            chosen_set.add(p1)
            pair_count += 1
            if pair_count == need_pairs:
                break

        if pair_count != need_pairs:
            continue
        if len({r for r, _ in chosen}) < 2 or len({c for _, c in chosen}) < 2:
            continue
        return chosen

    return None


def _grid_size_for_objects(object_count: int) -> tuple[int, int]:
    if object_count == 2:
        h = int(_RNG.integers(18, 25))
        w = int(_RNG.integers(18, 25))
    else:
        h = int(_RNG.integers(21, 29))
        w = int(_RNG.integers(21, 29))
    return h, w


def _candidate_centers(height: int, width: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(SCATTER_RADIUS, height - SCATTER_RADIUS)
        for c in range(SCATTER_RADIUS, width - SCATTER_RADIUS)
    ]


def _place_centers(objects: list[dict[str, int]], height: int, width: int, want_overlap: bool) -> bool:
    centers = _candidate_centers(height, width)
    _RNG.shuffle(centers)

    for idx, obj in enumerate(objects):
        candidates: list[tuple[tuple[int, int], tuple[int, int, int, int]]] = []
        rect_h = obj["rect_h"]
        rect_w = obj["rect_w"]

        for center_r, center_c in centers:
            bbox = _rect_from_center(center_r, center_c, rect_h, rect_w)
            top, left, bottom, right = bbox
            if not (0 <= top <= bottom < height and 0 <= left <= right < width):
                continue

            if idx == 0:
                candidates.append(((center_r, center_c), bbox))
                continue

            if want_overlap and idx == 1:
                first_bbox = objects[0]["bbox"]
                if not _bbox_intersects(bbox, first_bbox):
                    continue
                if _bbox_intersection_area(bbox, first_bbox) >= min(obj["count"], objects[0]["count"]):
                    continue
                candidates.append(((center_r, center_c), bbox))
                continue

            ok = True
            for prev in objects[:idx]:
                if _bbox_intersects(_expand_bbox(bbox, 1), _expand_bbox(prev["bbox"], 1)):
                    ok = False
                    break
            if ok:
                candidates.append(((center_r, center_c), bbox))

        if not candidates:
            return False

        center, bbox = candidates[int(_RNG.integers(len(candidates)))]
        obj["center_r"], obj["center_c"] = center
        obj["bbox"] = bbox

    return True


def _has_any_overlap(objects: list[dict[str, int]]) -> bool:
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if _bbox_intersects(objects[i]["bbox"], objects[j]["bbox"]):
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    object_count = int(_RNG.choice(OBJECT_COUNT_OPTIONS))
    height, width = _grid_size_for_objects(object_count)
    counts = sorted(int(x) for x in _RNG.choice(COUNT_OPTIONS, size=object_count, replace=False))
    colors = [int(x) for x in _RNG.choice(PALETTE, size=object_count, replace=False)]
    want_overlap = object_count >= 2 and bool(_RNG.random() < OVERLAP_PROB)

    objects: list[dict[str, int]] = []
    for order, (count, color) in enumerate(zip(counts, colors)):
        rect_h, rect_w = _best_factor_pair(count)
        objects.append(
            {
                "order": order,
                "color": color,
                "count": count,
                "rect_h": rect_h,
                "rect_w": rect_w,
            }
        )

    if not _place_centers(objects, height, width, want_overlap):
        return None

    has_overlap = _has_any_overlap(objects)
    if want_overlap != has_overlap:
        return None

    grid = np.full((height, width), BG, dtype=int)
    occupied: set[tuple[int, int]] = set()

    for obj in sorted(objects, key=lambda item: item["count"], reverse=True):
        center = (obj["center_r"], obj["center_c"])
        points = _scatter_points(height, width, center, obj["count"], occupied)
        if points is None:
            return None

        top, left, bottom, right = obj["bbox"]
        if all(top <= r <= bottom and left <= c <= right for r, c in points):
            return None

        obj["points"] = points
        for r, c in points:
            grid[r, c] = obj["color"]
            occupied.add((r, c))

    expected = _render_groups((height, width), objects)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
