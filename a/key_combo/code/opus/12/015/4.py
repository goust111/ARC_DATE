import itertools
import random

import numpy as np


RECTANGLE_COLORS = (1, 2, 3, 4, 5)
NOISE_COLORS = (6, 7, 8, 9)

GEN_MAX_ATTEMPTS = 800
RECTANGLE_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 160


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_dims(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _border_points(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        pts.append((r, c1))
    return pts


def _border_set(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    return set(_border_points(bbox))


def _inside_bbox(r: int, c: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    rectangles: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != 0):
        coords = np.argwhere(grid == int(color))
        if coords.shape[0] < 8:
            continue

        bbox = _bbox_from_coords(coords)
        height, width = _bbox_dims(bbox)
        short_side, long_side = sorted((height, width))
        if not (4 <= short_side <= 7 and 5 <= long_side <= 9):
            continue

        r0, c0, r1, c1 = bbox
        if not np.all(
            (coords[:, 0] == r0)
            | (coords[:, 0] == r1)
            | (coords[:, 1] == c0)
            | (coords[:, 1] == c1)
        ):
            continue

        if not (
            np.any(coords[:, 0] == r0)
            and np.any(coords[:, 0] == r1)
            and np.any(coords[:, 1] == c0)
            and np.any(coords[:, 1] == c1)
        ):
            continue

        rectangles.append({"color": int(color), "bbox": bbox})

    assert 3 <= len(rectangles) <= 5, "expected 3-5 hollow rectangles"
    return rectangles


def _pair_top_color(grid: np.ndarray, rect_a: dict, rect_b: dict) -> int:
    shared = _border_set(rect_a["bbox"]) & _border_set(rect_b["bbox"])
    assert shared, "every rectangle pair must intersect on the border"

    visible = {
        int(grid[r, c])
        for r, c in shared
        if int(grid[r, c]) in (rect_a["color"], rect_b["color"])
    }
    assert len(visible) == 1, "pairwise occlusion must reveal a unique visible top color"
    return visible.pop()


def _bottom_to_top_order(rectangles: list[dict], grid: np.ndarray) -> list[int]:
    edges = {rect["color"]: set() for rect in rectangles}
    indegree = {rect["color"]: 0 for rect in rectangles}

    for rect_a, rect_b in itertools.combinations(rectangles, 2):
        top = _pair_top_color(grid, rect_a, rect_b)
        bottom = rect_b["color"] if top == rect_a["color"] else rect_a["color"]
        if top not in edges[bottom]:
            edges[bottom].add(top)
            indegree[top] += 1

    order: list[int] = []
    while indegree:
        candidates = sorted(color for color, deg in indegree.items() if deg == 0)
        assert len(candidates) == 1, "occlusion comparisons must imply a unique total order"
        color = candidates[0]
        order.append(color)
        del indegree[color]
        for above in edges[color]:
            if above in indegree:
                indegree[above] -= 1

    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the bottom-to-top stacking order of overlapping hollow rectangles into a one-row color strip.

    Input structure:
    - The grid has a black background with 3-5 axis-aligned hollow rectangles in distinct colors.
    - Rectangle borders are one cell thick, their interiors are black, and some border crossings are occluded by higher rectangles.
    - A few isolated single-pixel noise dots may appear on the background but do not belong to any rectangle.

    Rule:
    - Ignore isolated noise pixels.
    - Recover each hollow rectangle from its visible border fragments and infer its full bounding box.
    - For every rectangle pair, use the visible color at their shared border-crossing cells to decide which rectangle is above the other.
    - Output a 1xN grid listing rectangle colors from bottom layer to top layer.

    Key insight:
    - The answer is not encoded by position or size; it is encoded only by which color remains visible at pairwise border crossings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    rectangles = _extract_rectangles(g)
    order = _bottom_to_top_order(rectangles, g)
    return np.array([order], dtype=int)


def _draw_rectangles(
    shape: tuple[int, int],
    rectangles: list[dict],
    bottom_to_top: list[int],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    rank = {color: idx for idx, color in enumerate(bottom_to_top)}
    for rect in sorted(rectangles, key=lambda item: rank[item["color"]]):
        for r, c in _border_points(rect["bbox"]):
            grid[r, c] = rect["color"]
    return grid


def _intersection_geometry_ok(bboxes: list[tuple[int, int, int, int]]) -> bool:
    border_sets = [_border_set(bbox) for bbox in bboxes]
    for i, j in itertools.combinations(range(len(bboxes)), 2):
        shared = border_sets[i] & border_sets[j]
        if not shared or len(shared) > 4:
            return False
        for r, c in shared:
            if (r + 1, c) in shared or (r, c + 1) in shared:
                return False
        if not any(sum((r, c) in border for border in border_sets) == 2 for r, c in shared):
            return False
    return True


def _visible_fraction(grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]) -> float:
    visible = int(np.count_nonzero(grid == int(color)))
    return visible / len(_border_points(bbox))


def _easy_geometry_orders(rectangles: list[dict]) -> list[list[int]]:
    def order_by(key):
        return [rect["color"] for rect in sorted(rectangles, key=key)]

    return [
        order_by(lambda rect: rect["bbox"][1]),
        order_by(lambda rect: rect["bbox"][0]),
        order_by(lambda rect: _bbox_dims(rect["bbox"])[0] * _bbox_dims(rect["bbox"])[1]),
        order_by(lambda rect: _bbox_dims(rect["bbox"])[1]),
        order_by(lambda rect: _bbox_dims(rect["bbox"])[0]),
    ]


def _shortcut_order(bottom_to_top: list[int], rectangles: list[dict]) -> bool:
    for easy in _easy_geometry_orders(rectangles):
        if bottom_to_top == easy or bottom_to_top == list(reversed(easy)):
            return True
    return False


def _place_noise(
    grid: np.ndarray,
    rectangles: list[dict],
    target_count: int,
) -> np.ndarray:
    out = grid.copy()
    palette = list(NOISE_COLORS)
    random.shuffle(palette)

    def isolated_zero(r: int, c: int) -> bool:
        if out[r, c] != 0:
            return False
        for bbox in (rect["bbox"] for rect in rectangles):
            if _inside_bbox(r, c, bbox):
                return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and out[rr, cc] != 0:
                    return False
        return True

    placed = 0
    for color in palette:
        if placed >= target_count:
            break
        candidates = [
            (r, c)
            for r in range(out.shape[0])
            for c in range(out.shape[1])
            if isolated_zero(r, c)
        ]
        if not candidates:
            break
        r, c = random.choice(candidates)
        out[r, c] = int(color)
        placed += 1

    return out


def _dimensions_for_count(count: int) -> tuple[int, int]:
    if count == 3:
        return random.randint(14, 18), random.randint(14, 18)
    if count == 4:
        return random.randint(16, 21), random.randint(16, 21)
    return random.randint(20, 24), random.randint(20, 24)


def _sample_rectangles(count: int, height: int, width: int) -> list[tuple[int, int, int, int]] | None:
    for _ in range(RECTANGLE_ATTEMPTS):
        anchor_r = random.randint(max(4, height // 3), min(height - 5, (2 * height) // 3))
        anchor_c = random.randint(max(4, width // 3), min(width - 5, (2 * width) // 3))

        bboxes: list[tuple[int, int, int, int]] = []
        used_sizes: set[tuple[int, int]] = set()
        used_boxes: set[tuple[int, int, int, int]] = set()
        for _ in range(count):
            chosen = None
            for _ in range(PLACEMENT_ATTEMPTS):
                box_h = random.randint(4, 8)
                box_w = random.randint(5, 9)
                short_side, long_side = sorted((box_h, box_w))
                if not (4 <= short_side <= 7 and 5 <= long_side <= 9):
                    continue
                if (box_h, box_w) in used_sizes:
                    continue

                r_min = max(1, anchor_r - box_h + 1)
                r_max = min(anchor_r - 1, height - box_h - 2)
                c_min = max(1, anchor_c - box_w + 1)
                c_max = min(anchor_c - 1, width - box_w - 2)
                if r_min > r_max or c_min > c_max:
                    continue

                r0 = random.randint(r_min, r_max)
                c0 = random.randint(c_min, c_max)
                bbox = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
                if bbox in used_boxes:
                    continue
                chosen = bbox
                break

            if chosen is None:
                break
            bboxes.append(chosen)
            used_boxes.add(chosen)
            used_sizes.add(_bbox_dims(chosen))

        if len(bboxes) != count:
            continue
        if _intersection_geometry_ok(bboxes):
            return bboxes
    return None


def _validate_instance(
    grid: np.ndarray,
    rectangles: list[dict],
    bottom_to_top: list[int],
) -> bool:
    extracted = _extract_rectangles(grid)
    truth = sorted((rect["color"], rect["bbox"]) for rect in rectangles)
    recovered = sorted((rect["color"], rect["bbox"]) for rect in extracted)
    if recovered != truth:
        return False

    if _bottom_to_top_order(extracted, grid) != bottom_to_top:
        return False

    rank = {color: idx for idx, color in enumerate(bottom_to_top)}
    for rect_a, rect_b in itertools.combinations(rectangles, 2):
        visible_top = _pair_top_color(grid, rect_a, rect_b)
        expected_top = rect_a["color"] if rank[rect_a["color"]] > rank[rect_b["color"]] else rect_b["color"]
        if visible_top != expected_top:
            return False

    max_density = 0.18 if len(rectangles) <= 4 else 0.24
    if np.count_nonzero(grid) > max_density * grid.size:
        return False

    min_fraction = 0.55 if len(rectangles) <= 4 else 0.42
    for rect in rectangles:
        if _visible_fraction(grid, rect["color"], rect["bbox"]) < min_fraction:
            return False

    expected = np.array([bottom_to_top], dtype=int)
    return np.array_equal(main(grid), expected)


def _gen_attempt() -> np.ndarray | None:
    count = random.choices([3, 4, 5], weights=[4, 4, 2], k=1)[0]
    height, width = _dimensions_for_count(count)
    bboxes = _sample_rectangles(count, height, width)
    if bboxes is None:
        return None

    colors = random.sample(list(RECTANGLE_COLORS), count)
    rectangles = [{"color": color, "bbox": bbox} for color, bbox in zip(colors, bboxes)]
    bottom_to_top = colors[:]
    random.shuffle(bottom_to_top)
    if _shortcut_order(bottom_to_top, rectangles):
        return None

    grid = _draw_rectangles((height, width), rectangles, bottom_to_top)
    noise_target = random.choice([0, 1, 2, 3 if count <= 4 else 2])
    grid = _place_noise(grid, rectangles, int(noise_target))

    if not _validate_instance(grid, rectangles, bottom_to_top):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlapping-rectangle input")
