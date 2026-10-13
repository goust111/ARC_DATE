import random

import numpy as np


# --------------- solver ---------------


def _extract_specs(grid: np.ndarray) -> list[dict]:
    specs: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == 0:
            continue
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        first_idx = int(np.lexsort((coords[:, 1], coords[:, 0]))[0])
        first = tuple(map(int, coords[first_idx]))
        specs.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "area": (r1 - r0 + 1) * (c1 - c0 + 1),
                "count": int(coords.shape[0]),
                "first": first,
            }
        )
    return specs


def _paint_specs(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    order = sorted(
        specs,
        key=lambda spec: (
            -int(spec["area"]),
            -int(spec["count"]),
            int(spec["first"][0]),
            int(spec["first"][1]),
            int(spec["bbox"][0]),
            int(spec["bbox"][1]),
        ),
    )
    for spec in order:
        r0, c0, r1, c1 = map(int, spec["bbox"])
        out[r0 : r1 + 1, c0 : c1 + 1] = int(spec["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered colored pixels by color, fill each color's bounding rectangle,
    and resolve rectangle overlaps by drawing larger rectangles first so smaller
    ones overwrite them. Equal-area overlaps are broken by input sparsity: the
    color with fewer source pixels is drawn later.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _paint_specs(g.shape, _extract_specs(g))


# --------------- generator ---------------

GRID_H_RANGE = (15, 29)
GRID_W_RANGE = (15, 29)
COLOR_COUNT_RANGE = (3, 5)
POINT_COUNT_RANGE = (2, 6)
RECT_CLEARANCE = 1
BOX_PLACEMENT_ATTEMPTS = 160
POINT_LAYOUT_ATTEMPTS = 40
POINT_SAMPLE_ATTEMPTS = 120
MAX_GENERATION_ATTEMPTS = 400

MODES = ("disjoint", "nested", "equal_area", "chain")
MODE_WEIGHTS = (18, 30, 24, 28)


def _candidate_axis(length: int) -> list[int]:
    return list(range(1, length - 1, 2))


def _make_box(
    rows: list[int],
    cols: list[int],
    r0i: int,
    c0i: int,
    rspan: int,
    cspan: int,
) -> dict:
    r1i = r0i + rspan - 1
    c1i = c0i + cspan - 1
    r0 = int(rows[r0i])
    c0 = int(cols[c0i])
    r1 = int(rows[r1i])
    c1 = int(cols[c1i])
    return {
        "r0i": int(r0i),
        "c0i": int(c0i),
        "r1i": int(r1i),
        "c1i": int(c1i),
        "bbox": (r0, c0, r1, c1),
        "area": (r1 - r0 + 1) * (c1 - c0 + 1),
        "capacity": rspan * cspan,
    }


def _random_box(
    rows: list[int],
    cols: list[int],
    rng: random.Random,
    min_rspan: int,
    max_rspan: int,
    min_cspan: int,
    max_cspan: int,
) -> dict | None:
    max_rspan = min(max_rspan, len(rows))
    max_cspan = min(max_cspan, len(cols))
    if min_rspan > max_rspan or min_cspan > max_cspan:
        return None
    rspan = rng.randint(min_rspan, max_rspan)
    cspan = rng.randint(min_cspan, max_cspan)
    r0i = rng.randint(0, len(rows) - rspan)
    c0i = rng.randint(0, len(cols) - cspan)
    return _make_box(rows, cols, r0i, c0i, rspan, cspan)


def _boxes_overlap(a: dict, b: dict) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a["bbox"])
    br0, bc0, br1, bc1 = map(int, b["bbox"])
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _boxes_close(a: dict, b: dict, margin: int = RECT_CLEARANCE) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a["bbox"])
    br0, bc0, br1, bc1 = map(int, b["bbox"])
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _span_lengths(box: dict) -> tuple[int, int]:
    return (
        int(box["r1i"]) - int(box["r0i"]) + 1,
        int(box["c1i"]) - int(box["c0i"]) + 1,
    )


def _sample_inner_box(
    container: dict,
    rows: list[int],
    cols: list[int],
    rng: random.Random,
    min_rspan: int,
    min_cspan: int,
) -> dict | None:
    outer_rspan, outer_cspan = _span_lengths(container)
    max_rspan = outer_rspan - 1
    max_cspan = outer_cspan - 1
    if min_rspan > max_rspan or min_cspan > max_cspan:
        return None
    for _ in range(BOX_PLACEMENT_ATTEMPTS):
        rspan = rng.randint(min_rspan, max_rspan)
        cspan = rng.randint(min_cspan, max_cspan)
        r0i = rng.randint(int(container["r0i"]), int(container["r1i"]) - rspan + 1)
        c0i = rng.randint(int(container["c0i"]), int(container["c1i"]) - cspan + 1)
        box = _make_box(rows, cols, r0i, c0i, rspan, cspan)
        if box["bbox"] != container["bbox"]:
            return box
    return None


def _place_disjoint_box(
    existing: list[dict],
    rows: list[int],
    cols: list[int],
    rng: random.Random,
) -> dict | None:
    for _ in range(BOX_PLACEMENT_ATTEMPTS):
        box = _random_box(rows, cols, rng, 2, 4, 2, 4)
        if box is None:
            return None
        if all(not _boxes_close(box, other) for other in existing):
            return box
    return None


def _complete_with_disjoint_boxes(
    boxes: list[dict],
    target: int,
    rows: list[int],
    cols: list[int],
    rng: random.Random,
) -> list[dict] | None:
    result = list(boxes)
    while len(result) < target:
        extra = _place_disjoint_box(result, rows, cols, rng)
        if extra is None:
            return None
        result.append(extra)
    return result


def _gen_disjoint_boxes(
    n_colors: int,
    rows: list[int],
    cols: list[int],
    rng: random.Random,
) -> tuple[list[dict], list[tuple[int, int]]] | None:
    boxes: list[dict] = []
    for _ in range(n_colors):
        box = _place_disjoint_box(boxes, rows, cols, rng)
        if box is None:
            return None
        boxes.append(box)
    return boxes, []


def _gen_nested_boxes(
    n_colors: int,
    rows: list[int],
    cols: list[int],
    rng: random.Random,
) -> tuple[list[dict], list[tuple[int, int]]] | None:
    for _ in range(BOX_PLACEMENT_ATTEMPTS):
        outer = _random_box(rows, cols, rng, 3, 5, 3, 5)
        if outer is None:
            return None
        inner = _sample_inner_box(outer, rows, cols, rng, 2, 2)
        if inner is None:
            continue
        boxes = _complete_with_disjoint_boxes([outer, inner], n_colors, rows, cols, rng)
        if boxes is not None:
            return boxes, []
    return None


def _gen_equal_area_boxes(
    n_colors: int,
    rows: list[int],
    cols: list[int],
    rng: random.Random,
) -> tuple[list[dict], list[tuple[int, int]]] | None:
    row_count = len(rows)
    col_count = len(cols)
    max_rspan = min(4, row_count)
    max_cspan = min(4, col_count)
    if max_rspan < 2 or max_cspan < 2:
        return None

    for _ in range(BOX_PLACEMENT_ATTEMPTS):
        rspan = rng.randint(2, max_rspan)
        cspan = rng.randint(2, max_cspan)
        offsets = [
            (dr, dc)
            for dr in range(-(rspan - 1), rspan)
            for dc in range(-(cspan - 1), cspan)
            if not (dr == 0 and dc == 0)
        ]
        rng.shuffle(offsets)
        for dr, dc in offsets:
            r0_lo = max(0, -dr)
            r0_hi = min(row_count - rspan, row_count - rspan - dr)
            c0_lo = max(0, -dc)
            c0_hi = min(col_count - cspan, col_count - cspan - dc)
            if r0_lo > r0_hi or c0_lo > c0_hi:
                continue
            ar0i = rng.randint(r0_lo, r0_hi)
            ac0i = rng.randint(c0_lo, c0_hi)
            a = _make_box(rows, cols, ar0i, ac0i, rspan, cspan)
            b = _make_box(rows, cols, ar0i + dr, ac0i + dc, rspan, cspan)
            if not _boxes_overlap(a, b) or a["bbox"] == b["bbox"]:
                continue
            boxes = _complete_with_disjoint_boxes([a, b], n_colors, rows, cols, rng)
            if boxes is not None:
                return boxes, [(0, 1)]
    return None


def _gen_chain_boxes(
    n_colors: int,
    rows: list[int],
    cols: list[int],
    rng: random.Random,
) -> tuple[list[dict], list[tuple[int, int]]] | None:
    for _ in range(BOX_PLACEMENT_ATTEMPTS):
        outer = _random_box(rows, cols, rng, 4, 6, 4, 6)
        if outer is None:
            return None
        middle = _sample_inner_box(outer, rows, cols, rng, 3, 3)
        if middle is None:
            continue
        inner = _sample_inner_box(middle, rows, cols, rng, 2, 2)
        if inner is None:
            continue
        boxes = _complete_with_disjoint_boxes([outer, middle, inner], n_colors, rows, cols, rng)
        if boxes is not None:
            return boxes, []
    return None


def _choose_boxes(
    mode: str,
    n_colors: int,
    rows: list[int],
    cols: list[int],
    rng: random.Random,
) -> tuple[list[dict], list[tuple[int, int]]] | None:
    if mode == "disjoint":
        return _gen_disjoint_boxes(n_colors, rows, cols, rng)
    if mode == "nested":
        return _gen_nested_boxes(n_colors, rows, cols, rng)
    if mode == "equal_area":
        return _gen_equal_area_boxes(n_colors, rows, cols, rng)
    if mode == "chain":
        return _gen_chain_boxes(n_colors, rows, cols, rng)
    raise ValueError(f"unknown mode: {mode}")


def _assign_counts(
    boxes: list[dict],
    tied_pairs: list[tuple[int, int]],
    rng: random.Random,
) -> bool:
    for box in boxes:
        hi = min(POINT_COUNT_RANGE[1], int(box["capacity"]))
        if hi < POINT_COUNT_RANGE[0]:
            return False
        box["count"] = rng.randint(POINT_COUNT_RANGE[0], hi)

    for i, j in tied_pairs:
        hi_i = min(POINT_COUNT_RANGE[1], int(boxes[i]["capacity"]))
        hi_j = min(POINT_COUNT_RANGE[1], int(boxes[j]["capacity"]))
        options = [
            (a, b)
            for a in range(POINT_COUNT_RANGE[0], hi_i + 1)
            for b in range(POINT_COUNT_RANGE[0], hi_j + 1)
            if a != b
        ]
        if not options:
            return False
        boxes[i]["count"], boxes[j]["count"] = rng.choice(options)

    return True


def _candidate_points(box: dict, rows: list[int], cols: list[int]) -> list[tuple[int, int]]:
    points = []
    for ri in range(int(box["r0i"]), int(box["r1i"]) + 1):
        for ci in range(int(box["c0i"]), int(box["c1i"]) + 1):
            points.append((int(rows[ri]), int(cols[ci])))
    return points


def _bbox_from_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [p[0] for p in points]
    cs = [p[1] for p in points]
    return min(rs), min(cs), max(rs), max(cs)


def _sample_points_for_box(
    box: dict,
    rows: list[int],
    cols: list[int],
    occupied: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    candidates = [p for p in _candidate_points(box, rows, cols) if p not in occupied]
    count = int(box["count"])
    if len(candidates) < count:
        return None

    r0, c0, r1, c1 = map(int, box["bbox"])
    target_bbox = (r0, c0, r1, c1)
    candidate_set = set(candidates)
    tl = (r0, c0)
    tr = (r0, c1)
    bl = (r1, c0)
    br = (r1, c1)

    if count == 2:
        pairs = []
        if tl in candidate_set and br in candidate_set:
            pairs.append([tl, br])
        if tr in candidate_set and bl in candidate_set:
            pairs.append([tr, bl])
        if pairs:
            return list(rng.choice(pairs))

    top = [p for p in candidates if p[0] == r0]
    bottom = [p for p in candidates if p[0] == r1]
    left = [p for p in candidates if p[1] == c0]
    right = [p for p in candidates if p[1] == c1]
    buckets = (top, bottom, left, right)
    if any(not bucket for bucket in buckets):
        return None

    for _ in range(POINT_SAMPLE_ATTEMPTS):
        chosen: list[tuple[int, int]] = []
        ok = True
        for bucket in buckets:
            if any(p in bucket for p in chosen):
                continue
            options = [p for p in bucket if p not in chosen]
            if not options:
                ok = False
                break
            chosen.append(rng.choice(options))
            if len(chosen) > count:
                ok = False
                break
        if not ok:
            continue

        remaining = [p for p in candidates if p not in chosen]
        rng.shuffle(remaining)
        chosen = chosen + remaining[: count - len(chosen)]
        if len(chosen) != count:
            continue
        if _bbox_from_points(chosen) == target_bbox:
            return chosen

    return None


def _overlap_degree(index: int, boxes: list[dict]) -> int:
    return sum(1 for j, other in enumerate(boxes) if j != index and _boxes_overlap(boxes[index], other))


def _assign_points(
    boxes: list[dict],
    rows: list[int],
    cols: list[int],
    rng: random.Random,
) -> list[list[tuple[int, int]]] | None:
    for _ in range(POINT_LAYOUT_ATTEMPTS):
        order = sorted(
            range(len(boxes)),
            key=lambda idx: (
                -_overlap_degree(idx, boxes),
                int(boxes[idx]["capacity"]),
                -int(boxes[idx]["count"]),
                rng.random(),
            ),
        )
        occupied: set[tuple[int, int]] = set()
        result: list[list[tuple[int, int]] | None] = [None] * len(boxes)
        ok = True
        for idx in order:
            pts = _sample_points_for_box(boxes[idx], rows, cols, occupied, rng)
            if pts is None:
                ok = False
                break
            result[idx] = pts
            occupied.update(pts)
        if ok:
            return [pts for pts in result if pts is not None]
    return None


def _has_cross_color_adjacency(grid: np.ndarray) -> bool:
    h, w = grid.shape
    for r, c in np.argwhere(grid != 0):
        color = int(grid[r, c])
        for nr in range(max(0, int(r) - 1), min(h, int(r) + 2)):
            for nc in range(max(0, int(c) - 1), min(w, int(c) + 2)):
                if nr == int(r) and nc == int(c):
                    continue
                other = int(grid[nr, nc])
                if other != 0 and other != color:
                    return True
    return False


def _pairwise_order_unambiguous(specs: list[dict]) -> bool:
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            a = specs[i]
            b = specs[j]
            ar0, ac0, ar1, ac1 = map(int, a["bbox"])
            br0, bc0, br1, bc1 = map(int, b["bbox"])
            overlap = not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)
            if overlap and int(a["area"]) == int(b["area"]) and int(a["count"]) == int(b["count"]):
                return False
    return True


def _build_specs(boxes: list[dict], points_by_box: list[list[tuple[int, int]]]) -> list[dict]:
    specs: list[dict] = []
    for box, points in zip(boxes, points_by_box):
        first = min(points)
        specs.append(
            {
                "color": int(box["color"]),
                "bbox": tuple(map(int, box["bbox"])),
                "area": int(box["area"]),
                "count": int(box["count"]),
                "first": first,
            }
        )
    return specs


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_colors = rng.randint(*COLOR_COUNT_RANGE)
    mode = rng.choices(MODES, weights=MODE_WEIGHTS, k=1)[0]

    min_h = GRID_H_RANGE[0] + 2 * max(0, n_colors - 3)
    min_w = GRID_W_RANGE[0] + 2 * max(0, n_colors - 3)
    if mode in {"nested", "chain"}:
        min_h += 2
        min_w += 2

    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    h = rng.randint(min_h, GRID_H_RANGE[1])
    w = rng.randint(min_w, GRID_W_RANGE[1])
    rows = _candidate_axis(h)
    cols = _candidate_axis(w)
    if len(rows) < 6 or len(cols) < 6:
        return None

    chosen = _choose_boxes(mode, n_colors, rows, cols, rng)
    if chosen is None:
        return None
    boxes, tied_pairs = chosen
    if len(boxes) != n_colors:
        return None
    if not _assign_counts(boxes, tied_pairs, rng):
        return None

    colors = rng.sample(list(range(1, 10)), n_colors)
    for box, color in zip(boxes, colors):
        box["color"] = int(color)

    points_by_box = _assign_points(boxes, rows, cols, rng)
    if points_by_box is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for box, points in zip(boxes, points_by_box):
        if len(points) != int(box["count"]):
            return None
        if _bbox_from_points(points) != tuple(map(int, box["bbox"])):
            return None
        for r, c in points:
            if grid[r, c] != 0:
                return None
            grid[r, c] = int(box["color"])

    specs = _build_specs(boxes, points_by_box)
    if not _pairwise_order_unambiguous(specs):
        return None
    if _has_cross_color_adjacency(grid):
        return None

    overlaps = sum(1 for i in range(len(boxes)) for j in range(i + 1, len(boxes)) if _boxes_overlap(boxes[i], boxes[j]))
    if mode == "disjoint" and overlaps != 0:
        return None
    if mode != "disjoint" and overlaps == 0:
        return None

    nonzero_colors = [int(color) for color in np.unique(grid) if int(color) != 0]
    if len(nonzero_colors) != n_colors:
        return None
    for color in nonzero_colors:
        count = int((grid == color).sum())
        if not (POINT_COUNT_RANGE[0] <= count <= POINT_COUNT_RANGE[1]):
            return None

    expected = _paint_specs(grid.shape, specs)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
