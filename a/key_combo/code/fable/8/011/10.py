from __future__ import annotations

import random
from collections import Counter

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9

    NOT_BLACK = [BLUE, RED, GREEN, YELLOW, GREY, PINK, ORANGE, TEAL, MAROON]


DIRS = ("up", "down", "left", "right")
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(coords)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _parse_rectangle_component(
    grid: np.ndarray, coords: list[tuple[int, int]], bg: int
) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert len(coords) == h * w and len(coords) > 1, "expected a solid rectangle component"

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(patch != bg), "rectangle bbox must be fully filled"

    colors = sorted(set(map(int, patch.flatten())))
    assert len(colors) == 2, "expected one body color and one indicator color"
    counts = {color: int(np.sum(patch == color)) for color in colors}

    singleton_colors = [color for color, count in counts.items() if count == 1]
    assert len(singleton_colors) == 1, "expected exactly one indicator pixel"
    indicator_color = singleton_colors[0]
    body_color = next(color for color in colors if color != indicator_color)

    rr, cc = map(int, np.argwhere(patch == indicator_color)[0])
    directions: list[str] = []
    if w % 2 == 1 and rr == 0 and cc == w // 2:
        directions.append("up")
    if w % 2 == 1 and rr == h - 1 and cc == w // 2:
        directions.append("down")
    if h % 2 == 1 and rr == h // 2 and cc == 0:
        directions.append("left")
    if h % 2 == 1 and rr == h // 2 and cc == w - 1:
        directions.append("right")
    assert len(directions) == 1, "indicator must sit at the center of exactly one side"

    return {
        "body_color": body_color,
        "indicator_color": indicator_color,
        "direction": directions[0],
        "bbox": (r0, c0, r1, c1),
        "indicator": (r0 + rr, c0 + cc),
    }


def _expanded_bbox(
    bbox: tuple[int, int, int, int], direction: str, dot: tuple[int, int]
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    dot_r, dot_c = dot
    if direction == "up":
        assert dot_r < r0
        return dot_r, c0, r1, c1
    if direction == "down":
        assert dot_r > r1
        return r0, c0, dot_r, c1
    if direction == "left":
        assert dot_c < c0
        return r0, dot_c, r1, c1
    assert direction == "right"
    assert dot_c > c1
    return r0, c0, r1, dot_c


def _front_indicator(
    bbox: tuple[int, int, int, int], direction: str
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if direction == "up":
        return r0, (c0 + c1) // 2
    if direction == "down":
        return r1, (c0 + c1) // 2
    if direction == "left":
        return (r0 + r1) // 2, c0
    return (r0 + r1) // 2, c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch each solid rectangle in the direction of its border indicator until the leading edge reaches the row or column encoded by the matching dot.

    Input structure:
    - The background is black.
    - Each non-singleton foreground component is a solid monochrome rectangle with one differently colored indicator pixel replacing the center cell of one side.
    - For every indicator color, there is one separate same-colored dot elsewhere on the grid.

    Rule:
    - For each rectangle, find the dot whose color matches the indicator pixel.
    - If the indicator is on the top or bottom side, extend the whole rectangle vertically until its front edge reaches the dot's row; if it is on the left or right side, extend horizontally until its front edge reaches the dot's column.
    - Erase the dots, keep each rectangle's body color, and move the indicator pixel with the advancing edge so it stays centered on that same side.

    Key insight:
    - The matching dot is not the destination cell: it only projects an invisible stopping line, so only one of the dot's coordinates matters.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    components = _components(g != bg)
    rectangles: list[dict[str, object]] = []
    indicator_positions: dict[int, set[tuple[int, int]]] = {}

    for coords in components:
        if len(coords) == 1:
            continue
        info = _parse_rectangle_component(g, coords, bg)
        rectangles.append(info)
        color = int(info["indicator_color"])
        indicator_positions.setdefault(color, set()).add(tuple(info["indicator"]))

    assert rectangles, "expected at least one rectangle component"

    dots_by_color: dict[int, tuple[int, int]] = {}
    out = g.copy()
    for color, indicator_cells in indicator_positions.items():
        all_cells = {tuple(map(int, rc)) for rc in np.argwhere(g == color)}
        dot_cells = list(all_cells - indicator_cells)
        assert len(dot_cells) == 1, "each indicator color must have exactly one matching dot"
        dots_by_color[color] = dot_cells[0]
        dr, dc = dot_cells[0]
        out[dr, dc] = bg

    for rect in rectangles:
        body_color = int(rect["body_color"])
        indicator_color = int(rect["indicator_color"])
        direction = str(rect["direction"])
        old_bbox = rect["bbox"]
        new_bbox = _expanded_bbox(old_bbox, direction, dots_by_color[indicator_color])
        nr0, nc0, nr1, nc1 = new_bbox
        out[nr0 : nr1 + 1, nc0 : nc1 + 1] = body_color
        ir, ic = _front_indicator(new_bbox, direction)
        out[ir, ic] = indicator_color

    return out


def _reserve_region(
    reserved: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int
) -> None:
    h, w = reserved.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _region_is_free(
    reserved: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int
) -> bool:
    h, w = reserved.shape
    if not (0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w):
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _sample_shape(direction: str, h: int, w: int) -> tuple[int, int, int] | None:
    if direction in ("up", "down"):
        widths = [x for x in (3, 5, 7) if x <= w - 4]
        if not widths:
            return None
        rect_w = random.choice(widths)
        rect_h = random.randint(2, min(5, h - 4))
        max_ext = min(7, h - rect_h - 2)
        if max_ext < 2:
            return None
        ext = random.randint(2, max_ext)
        return rect_h, rect_w, ext

    heights = [x for x in (3, 5, 7) if x <= h - 4]
    if not heights:
        return None
    rect_h = random.choice(heights)
    rect_w = random.randint(2, min(5, w - 4))
    max_ext = min(7, w - rect_w - 2)
    if max_ext < 2:
        return None
    ext = random.randint(2, max_ext)
    return rect_h, rect_w, ext


def _candidate_positions(
    canvas_h: int, canvas_w: int, box_h: int, box_w: int
) -> list[tuple[int, int]]:
    positions = [
        (r0, c0)
        for r0 in range(1, canvas_h - box_h)
        for c0 in range(1, canvas_w - box_w)
    ]
    random.shuffle(positions)
    return positions


def _pick_dot(
    reserved: np.ndarray, direction: str, bbox: tuple[int, int, int, int]
) -> tuple[int, int] | None:
    h, w = reserved.shape
    r0, c0, r1, c1 = bbox
    if direction in ("up", "down"):
        dot_r = r0 if direction == "up" else r1
        cols = list(range(w))
        random.shuffle(cols)
        for dot_c in cols:
            if c0 - 1 <= dot_c <= c1 + 1:
                continue
            if _region_is_free(reserved, dot_r, dot_c, dot_r, dot_c, pad=1):
                return dot_r, dot_c
        return None

    dot_c = c0 if direction == "left" else c1
    rows = list(range(h))
    random.shuffle(rows)
    for dot_r in rows:
        if r0 - 1 <= dot_r <= r1 + 1:
            continue
        if _region_is_free(reserved, dot_r, dot_c, dot_r, dot_c, pad=1):
            return dot_r, dot_c
    return None


def _input_bbox_from_output(
    out_bbox: tuple[int, int, int, int], direction: str, ext: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = out_bbox
    if direction == "up":
        return r0 + ext, c0, r1, c1
    if direction == "down":
        return r0, c0, r1 - ext, c1
    if direction == "left":
        return r0, c0 + ext, r1, c1
    return r0, c0, r1, c1 - ext


def _draw_input_object(
    grid: np.ndarray,
    body_color: int,
    indicator_color: int,
    direction: str,
    input_bbox: tuple[int, int, int, int],
    dot: tuple[int, int],
) -> None:
    r0, c0, r1, c1 = input_bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = body_color
    ir, ic = _front_indicator(input_bbox, direction)
    grid[ir, ic] = indicator_color
    dr, dc = dot
    grid[dr, dc] = indicator_color


def _expected_output(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, Color.BLACK, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["out_bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(spec["body_color"])
        ir, ic = spec["out_indicator"]
        out[ir, ic] = int(spec["indicator_color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(18, 30)
    grid_w = random.randint(18, 30)
    grid = np.full((grid_h, grid_w), Color.BLACK, dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    object_count = random.randint(2, 4)
    directions = random.sample(DIRS, k=object_count)
    palette = random.sample(Color.NOT_BLACK, k=2 * object_count)
    body_colors = palette[:object_count]
    indicator_colors = palette[object_count:]

    specs: list[dict[str, object]] = []
    for idx, direction in enumerate(directions):
        placed = False
        for _ in range(120):
            shape = _sample_shape(direction, grid_h, grid_w)
            if shape is None:
                return None
            rect_h, rect_w, ext = shape
            out_h = rect_h + ext if direction in ("up", "down") else rect_h
            out_w = rect_w if direction in ("up", "down") else rect_w + ext

            for r0, c0 in _candidate_positions(grid_h, grid_w, out_h, out_w):
                out_bbox = (r0, c0, r0 + out_h - 1, c0 + out_w - 1)
                if not _region_is_free(reserved, *out_bbox, pad=1):
                    continue
                dot = _pick_dot(reserved, direction, out_bbox)
                if dot is None:
                    continue

                input_bbox = _input_bbox_from_output(out_bbox, direction, ext)
                _draw_input_object(
                    grid,
                    body_colors[idx],
                    indicator_colors[idx],
                    direction,
                    input_bbox,
                    dot,
                )
                _reserve_region(reserved, *out_bbox, pad=1)
                _reserve_region(reserved, dot[0], dot[1], dot[0], dot[1], pad=1)
                specs.append(
                    {
                        "body_color": body_colors[idx],
                        "indicator_color": indicator_colors[idx],
                        "direction": direction,
                        "out_bbox": out_bbox,
                        "out_indicator": _front_indicator(out_bbox, direction),
                    }
                )
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    expected = _expected_output(grid.shape, specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    if len({spec["direction"] for spec in specs}) < min(2, object_count):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
