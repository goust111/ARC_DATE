from __future__ import annotations

from collections import Counter
import random

import numpy as np


MIN_RECT_SIDE = 3
MAX_RECT_SIDE = 9
GRID_SIDE_RANGE = (12, 20)
COUNT_VALUES = (4, 5, 6, 7)
GEN_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    return Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0]


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _object_from_pixels(color: int, coords: np.ndarray) -> dict[str, object]:
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    first = tuple(map(int, coords[int(order[0])]))
    bbox = (r0, c0, r1, c1)
    return {
        "color": int(color),
        "count": int(coords.shape[0]),
        "bbox": bbox,
        "area": int(_bbox_area(bbox)),
        "first": first,
    }


def _layer_sort_key(obj: dict[str, object]) -> tuple[int, int, int, int, int, int, int, int]:
    r0, c0, r1, c1 = obj["bbox"]  # type: ignore[misc]
    fr, fc = obj["first"]  # type: ignore[misc]
    return (
        -int(obj["count"]),
        -int(obj["area"]),
        int(fr),
        int(fc),
        int(r0),
        int(c0),
        int(r1),
        int(c1),
    )


def _render(shape: tuple[int, int], bg: int, objects: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for obj in sorted(objects, key=_layer_sort_key):
        r0, c0, r1, c1 = obj["bbox"]  # type: ignore[misc]
        out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(obj["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each foreground color's bounding box, layering by sample count.

    For every non-background color, infer the minimal axis-aligned bounding box
    that contains its scattered pixels, fill that box as a solid rectangle, and
    draw rectangles from largest pixel-count to smallest so sparser colors end
    up on top. Ties in pixel count are broken by larger bbox area first.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    color_order = [color for color in Counter(map(int, g.ravel())).keys() if color != int(bg)]

    objects: list[dict[str, object]] = []
    for color in color_order:
        coords = np.argwhere(g == int(color))
        if coords.size == 0:
            continue
        objects.append(_object_from_pixels(int(color), coords))

    return _render(g.shape, int(bg), objects)


def _sample_bbox_covering_core(
    rng: random.Random, height: int, width: int, core: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    cr0, cc0, cr1, cc1 = core
    min_h = max(MIN_RECT_SIDE, cr1 - cr0 + 1)
    min_w = max(MIN_RECT_SIDE, cc1 - cc0 + 1)
    max_h = min(MAX_RECT_SIDE, height)
    max_w = min(MAX_RECT_SIDE, width)
    if min_h > max_h or min_w > max_w:
        return None

    box_h = rng.randint(min_h, max_h)
    top_lo = max(0, cr1 - box_h + 1)
    top_hi = min(cr0, height - box_h)
    if top_lo > top_hi:
        return None
    top = rng.randint(top_lo, top_hi)

    box_w = rng.randint(min_w, max_w)
    left_lo = max(0, cc1 - box_w + 1)
    left_hi = min(cc0, width - box_w)
    if left_lo > left_hi:
        return None
    left = rng.randint(left_lo, left_hi)

    return (top, left, top + box_h - 1, left + box_w - 1)


def _random_bbox(rng: random.Random, height: int, width: int) -> tuple[int, int, int, int]:
    box_h = rng.randint(MIN_RECT_SIDE, min(MAX_RECT_SIDE, height))
    box_w = rng.randint(MIN_RECT_SIDE, min(MAX_RECT_SIDE, width))
    top = rng.randint(0, height - box_h)
    left = rng.randint(0, width - box_w)
    return (top, left, top + box_h - 1, left + box_w - 1)


def _make_hub_bboxes(
    rng: random.Random, height: int, width: int, n_rects: int
) -> list[tuple[int, int, int, int]] | None:
    max_core_h = min(3, height - 2)
    max_core_w = min(3, width - 2)
    core_h = rng.randint(1, max_core_h)
    core_w = rng.randint(1, max_core_w)
    core_r0 = rng.randint(1, height - core_h - 1)
    core_c0 = rng.randint(1, width - core_w - 1)
    core = (core_r0, core_c0, core_r0 + core_h - 1, core_c0 + core_w - 1)

    bboxes: list[tuple[int, int, int, int]] = []
    for _ in range(n_rects):
        for _ in range(80):
            bbox = _sample_bbox_covering_core(rng, height, width, core)
            if bbox is None or bbox in bboxes:
                continue
            bboxes.append(bbox)
            break
        else:
            return None
    return bboxes


def _make_chain_bboxes(
    rng: random.Random, height: int, width: int, n_rects: int
) -> list[tuple[int, int, int, int]] | None:
    bboxes = [_random_bbox(rng, height, width)]
    for _ in range(1, n_rects):
        prev = bboxes[-1]
        prev_h = prev[2] - prev[0] + 1
        prev_w = prev[3] - prev[1] + 1
        for _ in range(80):
            core_h = rng.randint(1, min(3, prev_h))
            core_w = rng.randint(1, min(3, prev_w))
            core_r0 = rng.randint(prev[0], prev[2] - core_h + 1)
            core_c0 = rng.randint(prev[1], prev[3] - core_w + 1)
            core = (core_r0, core_c0, core_r0 + core_h - 1, core_c0 + core_w - 1)
            bbox = _sample_bbox_covering_core(rng, height, width, core)
            if bbox is None or bbox in bboxes:
                continue
            if _intersection_area(prev, bbox) == 0:
                continue
            bboxes.append(bbox)
            break
        else:
            return None
    return bboxes


def _choose_counts(
    rng: random.Random, bboxes: list[tuple[int, int, int, int]], tie_case: bool
) -> list[int] | None:
    n = len(bboxes)
    areas = [_bbox_area(bbox) for bbox in bboxes]
    overlaps = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if _intersection_area(bboxes[i], bboxes[j]) > 0
    ]
    if not overlaps:
        return None

    if tie_case:
        tie_pairs = [(i, j) for i, j in overlaps if areas[i] != areas[j]]
        if not tie_pairs:
            return None
        i, j = rng.choice(tie_pairs)
        counts = [0] * n
        common = rng.choice((4, 5, 6) if n == 3 else COUNT_VALUES)
        counts[i] = common
        counts[j] = common
        remaining_indices = [idx for idx in range(n) if idx not in (i, j)]
        if remaining_indices:
            higher = [value for value in COUNT_VALUES if value > common]
            if not higher:
                return None
            counts[remaining_indices[0]] = rng.choice(higher)
        return counts

    values = rng.sample(list(COUNT_VALUES), n)
    rng.shuffle(values)
    return values


def _cover_edges(
    rng: random.Random,
    groups: dict[str, list[tuple[int, int]]],
    cover_map: dict[tuple[int, int], set[str]],
    limit: int,
) -> list[tuple[int, int]] | None:
    names = set(groups)

    def search(selected: list[tuple[int, int]], uncovered: set[str]) -> list[tuple[int, int]] | None:
        if not uncovered:
            return selected.copy()
        if len(selected) >= limit:
            return None

        for name in uncovered:
            if not any(cell not in selected for cell in groups[name]):
                return None

        group = min(uncovered, key=lambda name: sum(cell not in selected for cell in groups[name]))
        options = [cell for cell in groups[group] if cell not in selected]
        rng.shuffle(options)
        options.sort(key=lambda cell: -len(cover_map[cell] & uncovered))

        for cell in options:
            selected.append(cell)
            result = search(selected, uncovered - cover_map[cell])
            if result is not None:
                return result
            selected.pop()
        return None

    return search([], names)


def _choose_points_for_bbox(
    rng: random.Random,
    bbox: tuple[int, int, int, int],
    count: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = bbox
    cells = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in occupied
    ]
    if len(cells) < count:
        return None

    groups = {
        "top": [cell for cell in cells if cell[0] == r0],
        "bottom": [cell for cell in cells if cell[0] == r1],
        "left": [cell for cell in cells if cell[1] == c0],
        "right": [cell for cell in cells if cell[1] == c1],
    }
    if any(len(group) == 0 for group in groups.values()):
        return None

    cover_map: dict[tuple[int, int], set[str]] = {}
    for cell in cells:
        cover_map[cell] = {
            name
            for name, group_cells in groups.items()
            if cell in group_cells
        }

    selected = _cover_edges(rng, groups, cover_map, count)
    if selected is None:
        return None

    remaining = [cell for cell in cells if cell not in selected]
    while len(selected) < count:
        best_score = -1
        best_cells: list[tuple[int, int]] = []
        for cell in remaining:
            score = min(max(abs(cell[0] - other[0]), abs(cell[1] - other[1])) for other in selected)
            if score > best_score:
                best_score = score
                best_cells = [cell]
            elif score == best_score:
                best_cells.append(cell)
        if not best_cells:
            return None
        choice = rng.choice(best_cells)
        selected.append(choice)
        remaining.remove(choice)

    rng.shuffle(selected)
    return selected


def _build_scene(
    rng: random.Random, height: int, width: int, tie_case: bool
) -> list[dict[str, object]] | None:
    n_rects = rng.randint(2, 3) if tie_case else rng.randint(2, 4)
    layout = rng.choice(("hub", "chain"))
    if layout == "hub":
        bboxes = _make_hub_bboxes(rng, height, width, n_rects)
    else:
        bboxes = _make_chain_bboxes(rng, height, width, n_rects)
    if bboxes is None:
        return None

    counts = _choose_counts(rng, bboxes, tie_case)
    if counts is None:
        return None

    palette = list(range(10))
    bg = rng.choice(palette)
    palette.remove(bg)
    colors = rng.sample(palette, n_rects)

    occupied: set[tuple[int, int]] = set()
    order = sorted(range(n_rects), key=lambda idx: (_bbox_area(bboxes[idx]), idx))
    points_by_idx: dict[int, list[tuple[int, int]]] = {}
    for idx in order:
        points = _choose_points_for_bbox(rng, bboxes[idx], counts[idx], occupied)
        if points is None:
            return None
        points_by_idx[idx] = points
        occupied.update(points)

    objects: list[dict[str, object]] = []
    for idx in range(n_rects):
        coords = np.array(points_by_idx[idx], dtype=int)
        objects.append(
            {
                "color": int(colors[idx]),
                "count": int(counts[idx]),
                "bbox": tuple(map(int, bboxes[idx])),
                "area": int(_bbox_area(bboxes[idx])),
                "first": tuple(map(int, coords[np.lexsort((coords[:, 1], coords[:, 0]))[0]])),
                "points": points_by_idx[idx],
            }
        )

    overlap_pairs = [
        (i, j)
        for i in range(n_rects)
        for j in range(i + 1, n_rects)
        if _intersection_area(bboxes[i], bboxes[j]) > 0
    ]
    if not overlap_pairs:
        return None
    if tie_case:
        if not any(counts[i] == counts[j] and _bbox_area(bboxes[i]) != _bbox_area(bboxes[j]) for i, j in overlap_pairs):
            return None
    else:
        if len(set(counts)) != n_rects:
            return None

    objects.append({"bg": int(bg)})
    return objects


def _scene_to_input(shape: tuple[int, int], bg: int, objects: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for obj in objects:
        if "color" not in obj:
            continue
        for r, c in obj["points"]:  # type: ignore[misc]
            grid[int(r), int(c)] = int(obj["color"])
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_SIDE_RANGE)
    width = rng.randint(*GRID_SIDE_RANGE)
    tie_case = rng.random() < 0.5

    scene = _build_scene(rng, height, width, tie_case)
    if scene is None:
        return None

    bg = int(scene[-1]["bg"])
    objects = scene[:-1]
    inp = _scene_to_input((height, width), bg, objects)

    counts = Counter(map(int, inp.ravel()))
    if counts[int(bg)] <= max(count for color, count in counts.items() if color != int(bg)):
        return None

    expected = _render((height, width), bg, objects)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
