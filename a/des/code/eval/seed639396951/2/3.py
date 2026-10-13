from __future__ import annotations

import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
SIDES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 300
PLACEMENT_TRIES = 200


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _make_rect(r0: int, c0: int, h: int, w: int) -> dict[str, int | None]:
    return {
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r0 + h - 1),
        "c1": int(c0 + w - 1),
        "color": -1,
        "missing": None,
    }


def _area(rect: dict[str, int | None]) -> int:
    return (int(rect["r1"]) - int(rect["r0"]) + 1) * (int(rect["c1"]) - int(rect["c0"]) + 1)


def _bbox_intersects(a: dict[str, int | None], b: dict[str, int | None], pad: int = 0) -> bool:
    return not (
        int(a["r1"]) + pad < int(b["r0"])
        or int(b["r1"]) + pad < int(a["r0"])
        or int(a["c1"]) + pad < int(b["c0"])
        or int(b["c1"]) + pad < int(a["c0"])
    )


def _draw_full_border(grid: np.ndarray, rect: dict[str, int | None], color: int) -> None:
    r0, c0, r1, c1 = int(rect["r0"]), int(rect["c0"]), int(rect["r1"]), int(rect["c1"])
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _draw_input_border(grid: np.ndarray, rect: dict[str, int | None]) -> None:
    color = int(rect["color"])
    missing = rect["missing"]
    r0, c0, r1, c1 = int(rect["r0"]), int(rect["c0"]), int(rect["r1"]), int(rect["c1"])
    if missing != "top":
        grid[r0, c0 : c1 + 1] = color
    if missing != "bottom":
        grid[r1, c0 : c1 + 1] = color
    if missing != "left":
        grid[r0 : r1 + 1, c0] = color
    if missing != "right":
        grid[r0 : r1 + 1, c1] = color


def _paint_stripes(grid: np.ndarray, rect: dict[str, int | None], color: int) -> None:
    r0, c0, r1, c1 = int(rect["r0"]), int(rect["c0"]), int(rect["r1"]), int(rect["c1"])
    if r1 - r0 < 2 or c1 - c0 < 2:
        return
    rows = np.arange(r0 + 1, r1)[:, None] - r0
    cols = np.arange(c0 + 1, c1)[None, :] - c0
    stripe_mask = (rows + cols) % 2 == 0
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    interior[stripe_mask] = color


def _extract_rectangles(grid: np.ndarray, bg: int) -> list[dict[str, int | None]]:
    rects: list[dict[str, int | None]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        mask = grid == color
        r0, c0, r1, c1 = _bbox(mask)
        assert r1 - r0 >= 2 and c1 - c0 >= 2

        border = np.zeros_like(mask, dtype=bool)
        border[r0, c0 : c1 + 1] = True
        border[r1, c0 : c1 + 1] = True
        border[r0 : r1 + 1, c0] = True
        border[r0 : r1 + 1, c1] = True
        assert not np.any(mask & ~border)

        full_sides = 0
        full_sides += int(np.all(mask[r0, c0 : c1 + 1]))
        full_sides += int(np.all(mask[r1, c0 : c1 + 1]))
        full_sides += int(np.all(mask[r0 : r1 + 1, c0]))
        full_sides += int(np.all(mask[r0 : r1 + 1, c1]))
        assert 3 <= full_sides <= 4

        rects.append({"r0": r0, "c0": c0, "r1": r1, "c1": c1, "color": color, "missing": None})
    assert rects
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct each color's axis-aligned rectangle, complete one missing edge if needed,
    then fill the interior with diagonal stripes measured from the rectangle's top-left corner.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    rects = _extract_rectangles(g, bg)

    out = np.full_like(g, bg)
    for rect in sorted(rects, key=_area, reverse=True):
        _paint_stripes(out, rect, int(rect["color"]))
    for rect in rects:
        _draw_full_border(out, rect, int(rect["color"]))
    return out


def _render_input(shape: tuple[int, int], rects: list[dict[str, int | None]], bg: int = BG) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for rect in rects:
        _draw_input_border(grid, rect)
    return grid


def _render_output(shape: tuple[int, int], rects: list[dict[str, int | None]], bg: int = BG) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for rect in sorted(rects, key=_area, reverse=True):
        _paint_stripes(grid, rect, int(rect["color"]))
    for rect in rects:
        _draw_full_border(grid, rect, int(rect["color"]))
    return grid


def _sample_missing_states(count: int) -> list[str | None]:
    if count == 1:
        return [None if random.random() < 0.5 else random.choice(SIDES)]

    states = [None, random.choice(SIDES)]
    while len(states) < count:
        states.append(None if random.random() < 0.5 else random.choice(SIDES))
    random.shuffle(states)
    return states


def _attach_metadata(
    boxes: list[dict[str, int | None]], colors: list[int], missing_states: list[str | None]
) -> list[dict[str, int | None]]:
    rects: list[dict[str, int | None]] = []
    for box, color, missing in zip(boxes, colors, missing_states, strict=True):
        rect = dict(box)
        rect["color"] = int(color)
        rect["missing"] = missing
        rects.append(rect)
    return rects


def _sample_scatter_rect(shape: tuple[int, int], existing: list[dict[str, int | None]]) -> dict[str, int | None] | None:
    h, w = shape
    max_rect_h = min(11, h)
    max_rect_w = min(11, w)
    if max_rect_h < 4 or max_rect_w < 4:
        return None

    for _ in range(PLACEMENT_TRIES):
        rect_h = random.randint(4, max_rect_h)
        rect_w = random.randint(4, max_rect_w)
        r0 = random.randint(0, h - rect_h)
        c0 = random.randint(0, w - rect_w)
        rect = _make_rect(r0, c0, rect_h, rect_w)
        if all(not _bbox_intersects(rect, other, pad=1) for other in existing):
            return rect
    return None


def _sample_adjacent_rect(
    shape: tuple[int, int], anchor: dict[str, int | None]
) -> dict[str, int | None] | None:
    h, w = shape
    anchor_h = int(anchor["r1"]) - int(anchor["r0"]) + 1
    anchor_w = int(anchor["c1"]) - int(anchor["c0"]) + 1
    max_rect_h = min(11, h)
    max_rect_w = min(11, w)

    for _ in range(PLACEMENT_TRIES):
        rect_h = random.randint(4, max_rect_h)
        rect_w = random.randint(4, max_rect_w)
        orientation = random.choice(("horizontal", "vertical"))
        gap = random.randint(0, 2)

        if orientation == "horizontal":
            min_overlap = min(anchor_h, rect_h, 2)
            r_min = max(0, int(anchor["r0"]) - rect_h + min_overlap)
            r_max = min(h - rect_h, int(anchor["r1"]) - min_overlap + 1)
            if r_min > r_max:
                continue
            r0 = random.randint(r_min, r_max)
            if random.random() < 0.5:
                c0 = int(anchor["c1"]) + gap + 1
            else:
                c0 = int(anchor["c0"]) - gap - rect_w
            if not (0 <= c0 <= w - rect_w):
                continue
        else:
            min_overlap = min(anchor_w, rect_w, 2)
            c_min = max(0, int(anchor["c0"]) - rect_w + min_overlap)
            c_max = min(w - rect_w, int(anchor["c1"]) - min_overlap + 1)
            if c_min > c_max:
                continue
            c0 = random.randint(c_min, c_max)
            if random.random() < 0.5:
                r0 = int(anchor["r1"]) + gap + 1
            else:
                r0 = int(anchor["r0"]) - gap - rect_h
            if not (0 <= r0 <= h - rect_h):
                continue

        rect = _make_rect(r0, c0, rect_h, rect_w)
        if not _bbox_intersects(rect, anchor, pad=0):
            return rect
    return None


def _build_scatter_boxes(shape: tuple[int, int], count: int) -> list[dict[str, int | None]] | None:
    boxes: list[dict[str, int | None]] = []
    for _ in range(count):
        rect = _sample_scatter_rect(shape, boxes)
        if rect is None:
            return None
        boxes.append(rect)
    return boxes


def _build_adjacent_boxes(shape: tuple[int, int], count: int) -> list[dict[str, int | None]] | None:
    h, w = shape
    if h < 10 or w < 10 or count < 2:
        return None

    for _ in range(PLACEMENT_TRIES):
        first = _sample_scatter_rect(shape, [])
        if first is None:
            continue
        second = _sample_adjacent_rect(shape, first)
        if second is None:
            continue
        boxes = [first, second]
        ok = True
        for _ in range(count - 2):
            rect = _sample_scatter_rect(shape, boxes)
            if rect is None:
                ok = False
                break
            boxes.append(rect)
        if ok:
            return boxes
    return None


def _sample_inner_rect(outer: dict[str, int | None]) -> dict[str, int | None] | None:
    outer_h = int(outer["r1"]) - int(outer["r0"]) + 1
    outer_w = int(outer["c1"]) - int(outer["c0"]) + 1
    max_inner_h = outer_h - 4
    max_inner_w = outer_w - 4
    if max_inner_h < 4 or max_inner_w < 4:
        return None

    for _ in range(PLACEMENT_TRIES):
        inner_h = random.randint(4, max_inner_h)
        inner_w = random.randint(4, max_inner_w)
        r_min = int(outer["r0"]) + 2
        r_max = int(outer["r1"]) - inner_h - 1
        c_min = int(outer["c0"]) + 2
        c_max = int(outer["c1"]) - inner_w - 1
        if r_min > r_max or c_min > c_max:
            continue
        r0 = random.randint(r_min, r_max)
        c0 = random.randint(c_min, c_max)
        return _make_rect(r0, c0, inner_h, inner_w)
    return None


def _build_nested_boxes(shape: tuple[int, int], count: int) -> list[dict[str, int | None]] | None:
    h, w = shape
    if min(h, w) < 16 or count < 2:
        return None

    outer_max_h = min(16, h)
    outer_max_w = min(16, w)
    for _ in range(PLACEMENT_TRIES):
        outer_h = random.randint(8, outer_max_h)
        outer_w = random.randint(8, outer_max_w)
        r0 = random.randint(0, h - outer_h)
        c0 = random.randint(0, w - outer_w)
        outer = _make_rect(r0, c0, outer_h, outer_w)
        inner = _sample_inner_rect(outer)
        if inner is None:
            continue

        boxes = [outer, inner]
        ok = True
        for _ in range(count - 2):
            rect = _sample_scatter_rect(shape, boxes)
            if rect is None:
                ok = False
                break
            boxes.append(rect)
        if ok:
            return boxes
    return None


def _sample_layout() -> tuple[str, int, tuple[int, int]]:
    layout = random.choices(("scatter", "adjacent", "nested"), weights=(4, 3, 2), k=1)[0]
    if layout == "scatter":
        count = random.randint(1, 4)
        low, high = {1: (10, 20), 2: (12, 24), 3: (16, 28), 4: (20, 30)}[count]
    elif layout == "adjacent":
        count = random.randint(2, 4)
        low, high = {2: (12, 24), 3: (16, 28), 4: (20, 30)}[count]
    else:
        count = random.randint(2, 3)
        low, high = {2: (16, 30), 3: (18, 30)}[count]
    return layout, count, (random.randint(low, high), random.randint(low, high))


def _build_boxes(layout: str, shape: tuple[int, int], count: int) -> list[dict[str, int | None]] | None:
    if layout == "scatter":
        return _build_scatter_boxes(shape, count)
    if layout == "adjacent":
        return _build_adjacent_boxes(shape, count)
    return _build_nested_boxes(shape, count)


def _validate_case(inp: np.ndarray, expected: np.ndarray, rects: list[dict[str, int | None]]) -> bool:
    if not np.array_equal(main(inp), expected):
        return False
    if len({int(rect["color"]) for rect in rects}) != len(rects):
        return False
    for rect in rects:
        r0, c0, r1, c1 = int(rect["r0"]), int(rect["c0"]), int(rect["r1"]), int(rect["c1"])
        if r1 - r0 < 3 or c1 - c0 < 3:
            return False
    return True


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        layout, count, shape = _sample_layout()
        boxes = _build_boxes(layout, shape, count)
        if boxes is None:
            continue
        colors = random.sample(COLORS, count)
        missing_states = _sample_missing_states(count)
        rects = _attach_metadata(boxes, colors, missing_states)

        inp = _render_input(shape, rects)
        expected = _render_output(shape, rects)
        if _validate_case(inp, expected, rects):
            return inp, expected
    raise RuntimeError("failed to generate a valid ARC rectangle-stripe case")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
