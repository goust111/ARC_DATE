from __future__ import annotations

import itertools
import math
import random

import numpy as np


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[np.ndarray] = []

    for r0 in range(height):
        for c0 in range(width):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            components.append(np.asarray(coords, dtype=int))

    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _rect_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _touch_count(bbox: tuple[int, int, int, int], height: int, width: int) -> int:
    r0, c0, r1, c1 = bbox
    return int(r0 == 0) + int(c0 == 0) + int(r1 == height - 1) + int(c1 == width - 1)


def _layer_key(obj: dict[str, int | tuple[int, int, int, int]], height: int, width: int) -> tuple[int, int, int, int, int]:
    bbox = obj["bbox"]
    assert isinstance(bbox, tuple)
    return (
        _touch_count(bbox, height, width),
        -_rect_area(bbox),
        int(bbox[0]),
        int(bbox[1]),
        int(obj["color"]),
    )


def _render_rectangles(
    shape: tuple[int, int],
    bg: int,
    objects: list[dict[str, int | tuple[int, int, int, int]]],
    order: tuple[int, ...] | list[int],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for idx in order:
        bbox = objects[int(idx)]["bbox"]
        assert isinstance(bbox, tuple)
        r0, c0, r1, c1 = bbox
        out[r0 : r1 + 1, c0 : c1 + 1] = int(objects[int(idx)]["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Layer foreground objects by how many grid edges they touch.

    Input structure:
    - Background is the dominant color.
    - Each foreground color corresponds to one 4-connected object.
    - Some objects touch the outer border, some stay fully interior.

    Rule:
    - Count how many of the four grid edges each object touches.
    - Remove all foreground pixels.
    - Redraw objects from fewer touched edges to more touched edges.
    - For equal edge-touch counts, redraw larger objects first, i.e. sort by
      area descending. Later objects overwrite earlier ones.

    Key insight:
    - The visible colored remnants identify each object's original rectangle by
      its bounding box; layer order depends only on border contact count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    bg = _background_color(g)

    colors = [int(v) for v in np.unique(g) if int(v) != bg]
    if not colors:
        return g.copy()

    objects: list[dict[str, int | tuple[int, int, int, int]]] = []
    for color in sorted(colors):
        mask = g == color
        comps = _components4(mask)
        assert len(comps) == 1, "each visible object color must stay 4-connected"
        bbox = _bbox(comps[0])
        r0, c0, r1, c1 = bbox
        window = mask[r0 : r1 + 1, c0 : c1 + 1]
        assert window[0].any() and window[-1].any(), "object must expose top and bottom of its bbox"
        assert window[:, 0].any() and window[:, -1].any(), "object must expose left and right of its bbox"
        objects.append({"color": color, "bbox": bbox})

    objects.sort(key=lambda obj: _layer_key(obj, height, width))
    out = np.full_like(g, bg)
    for obj in objects:
        bbox = obj["bbox"]
        assert isinstance(bbox, tuple)
        r0, c0, r1, c1 = bbox
        out[r0 : r1 + 1, c0 : c1 + 1] = int(obj["color"])
    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (10, 18)
GRID_WIDTH_RANGE = (10, 18)
OBJECT_COUNT_RANGE = (3, 5)
TOUCH_CHOICES = (0, 1, 2, 3)
TOUCH_WEIGHTS = (3, 3, 2, 1)
DUPLICATE_TOUCH_PROB = 0.5
REQUIRE_CHANGE_PROB = 0.7
GEN_RETRY_LIMIT = 400
RECT_SAMPLE_LIMIT = 120
MAX_RECT_FRACTION = 0.32
MIN_VISIBLE_FRACTION = 0.35


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _sample_touch_counts(n_objects: int) -> list[int]:
    counts = [0, random.choice((1, 2, 3))]
    if n_objects >= 4 and random.random() < DUPLICATE_TOUCH_PROB:
        dup = random.choice((0, 1, 2))
        counts = [dup, dup, random.choice(TOUCH_CHOICES)]
    while len(counts) < n_objects:
        counts.append(random.choices(TOUCH_CHOICES, weights=TOUCH_WEIGHTS, k=1)[0])
    random.shuffle(counts)
    return counts[:n_objects]


def _sample_rectangle(height: int, width: int, touch_count: int) -> tuple[int, int, int, int] | None:
    max_area = max(12, int(height * width * MAX_RECT_FRACTION))

    for _ in range(RECT_SAMPLE_LIMIT):
        if touch_count == 0:
            rect_h = random.randint(2, min(6, height - 2))
            rect_w = random.randint(2, min(6, width - 2))
            r0 = random.randint(1, height - rect_h - 1)
            c0 = random.randint(1, width - rect_w - 1)
            bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        elif touch_count == 1:
            edge = random.choice(("top", "bottom", "left", "right"))
            if edge in ("top", "bottom"):
                rect_h = random.randint(2, min(6, height - 1))
                rect_w = random.randint(2, min(7, width - 2))
                r0 = 0 if edge == "top" else height - rect_h
                c0 = random.randint(1, width - rect_w - 1)
            else:
                rect_h = random.randint(2, min(7, height - 2))
                rect_w = random.randint(2, min(6, width - 1))
                r0 = random.randint(1, height - rect_h - 1)
                c0 = 0 if edge == "left" else width - rect_w
            bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        elif touch_count == 2:
            if random.random() < 0.55:
                vertical = random.choice(("top", "bottom"))
                horizontal = random.choice(("left", "right"))
                rect_h = random.randint(2, min(6, height - 1))
                rect_w = random.randint(2, min(6, width - 1))
                r0 = 0 if vertical == "top" else height - rect_h
                c0 = 0 if horizontal == "left" else width - rect_w
                bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            elif random.random() < 0.5:
                rect_w = random.randint(2, min(4, width - 2))
                c0 = random.randint(1, width - rect_w - 1)
                bbox = (0, c0, height - 1, c0 + rect_w - 1)
            else:
                rect_h = random.randint(2, min(4, height - 2))
                r0 = random.randint(1, height - rect_h - 1)
                bbox = (r0, 0, r0 + rect_h - 1, width - 1)
        elif touch_count == 3:
            if random.random() < 0.5:
                rect_h = random.randint(1, min(3, height - 1))
                r0 = 0 if random.random() < 0.5 else height - rect_h
                bbox = (r0, 0, r0 + rect_h - 1, width - 1)
            else:
                rect_w = random.randint(1, min(3, width - 1))
                c0 = 0 if random.random() < 0.5 else width - rect_w
                bbox = (0, c0, height - 1, c0 + rect_w - 1)
        else:
            return None

        area = _rect_area(bbox)
        if 4 <= area <= max_area:
            return bbox
    return None


def _visible_object_ok(
    grid: np.ndarray,
    bg: int,
    obj: dict[str, int | tuple[int, int, int, int]],
) -> bool:
    color = int(obj["color"])
    bbox = obj["bbox"]
    assert isinstance(bbox, tuple)
    mask = grid == color
    comps = _components4(mask)
    if len(comps) != 1:
        return False
    if _bbox(comps[0]) != bbox:
        return False

    r0, c0, r1, c1 = bbox
    window = mask[r0 : r1 + 1, c0 : c1 + 1]
    if not (window[0].any() and window[-1].any() and window[:, 0].any() and window[:, -1].any()):
        return False

    visible_area = int(mask.sum())
    min_visible = max(3, math.ceil(_rect_area(bbox) * MIN_VISIBLE_FRACTION))
    if visible_area < min_visible:
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    require_change = random.random() < REQUIRE_CHANGE_PROB

    palette = random.sample(list(range(10)), n_objects + 1)
    bg = palette[0]
    colors = palette[1:]

    touch_counts = _sample_touch_counts(n_objects)
    objects: list[dict[str, int | tuple[int, int, int, int]]] = []
    used_bboxes: set[tuple[int, int, int, int]] = set()

    for color, touch in zip(colors, touch_counts):
        bbox = None
        for _ in range(RECT_SAMPLE_LIMIT):
            candidate = _sample_rectangle(height, width, touch)
            if candidate is not None and candidate not in used_bboxes:
                bbox = candidate
                break
        if bbox is None:
            return None
        used_bboxes.add(bbox)
        objects.append({"color": int(color), "bbox": bbox})

    if len({_layer_key(obj, height, width) for obj in objects}) != len(objects):
        return None

    if not any(_overlaps(objects[i]["bbox"], objects[j]["bbox"]) for i in range(n_objects) for j in range(i + 1, n_objects)):
        return None

    target_order = tuple(sorted(range(n_objects), key=lambda idx: _layer_key(objects[idx], height, width)))
    target = _render_rectangles((height, width), bg, objects, target_order)

    same_candidates: list[np.ndarray] = []
    changed_candidates: list[np.ndarray] = []

    for order in itertools.permutations(range(n_objects)):
        grid = _render_rectangles((height, width), bg, objects, order)
        if _background_color(grid) != bg:
            continue
        if any(not _visible_object_ok(grid, bg, obj) for obj in objects):
            continue
        if np.array_equal(grid, target):
            same_candidates.append(grid)
        else:
            changed_candidates.append(grid)

    if require_change and changed_candidates:
        chosen = random.choice(changed_candidates)
    elif not require_change and same_candidates:
        chosen = random.choice(same_candidates)
    elif changed_candidates:
        chosen = random.choice(changed_candidates)
    elif same_candidates:
        chosen = random.choice(same_candidates)
    else:
        return None

    if not np.array_equal(main(chosen), target):
        return None
    return chosen


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
