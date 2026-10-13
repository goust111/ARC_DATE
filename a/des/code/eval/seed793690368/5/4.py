import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _paint_rect(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _legend_candidate(grid: np.ndarray, bg: int, row_index: int) -> dict | None:
    row = grid[row_index]
    cols = np.flatnonzero(row != bg)
    if cols.size < 3 or np.any(np.diff(cols) == 1):
        return None

    colors = [int(row[c]) for c in cols]
    if len(set(colors)) != len(colors):
        return None

    body = grid[1:] if row_index == 0 else grid[:-1]
    body_colors = set(map(int, body[body != bg]))
    if not body_colors or set(colors) != body_colors:
        return None

    neighbor_row = 1 if row_index == 0 else grid.shape[0] - 2
    isolation = int(np.sum(grid[neighbor_row, cols] == bg))
    return {"index": row_index, "colors": colors, "isolation": isolation}


def _find_legend(grid: np.ndarray, bg: int) -> dict:
    candidates = []
    for row_index in (0, grid.shape[0] - 1):
        cand = _legend_candidate(grid, bg, row_index)
        if cand is not None:
            candidates.append(cand)
    assert candidates, "expected a top or bottom legend row"
    best = max(candidates, key=lambda item: item["isolation"])
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each partially occluded solid rectangle from its visible extrema and
    redraw the full rectangles using the edge legend's left-to-right layer order.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    legend = _find_legend(g, bg)
    body = g[1:] if legend["index"] == 0 else g[:-1]

    colors = legend["colors"]
    assert set(colors) == set(map(int, body[body != bg]))

    out = np.full(body.shape, bg, dtype=int)
    for color in colors:
        coords = np.argwhere(body == color)
        assert coords.size, "each legend color must appear in the body"
        _paint_rect(out, _bbox(coords), int(color))
    return out


def _render_body(
    shape: tuple[int, int],
    bg: int,
    rects: list[tuple[int, int, int, int]],
    colors: list[int],
    order: list[int],
) -> np.ndarray:
    body = np.full(shape, bg, dtype=int)
    for idx in order:
        _paint_rect(body, rects[idx], colors[idx])
    return body


def _contains(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def _covered_by_other(
    r: int,
    c: int,
    rects: list[tuple[int, int, int, int]],
    skip: int,
) -> bool:
    for j, rect in enumerate(rects):
        if j == skip:
            continue
        r0, c0, r1, c1 = rect
        if r0 <= r <= r1 and c0 <= c <= c1:
            return True
    return False


def _side_exposed(rects: list[tuple[int, int, int, int]], idx: int) -> bool:
    r0, c0, r1, c1 = rects[idx]

    top_ok = any(not _covered_by_other(r0, c, rects, idx) for c in range(c0, c1 + 1))
    bottom_ok = any(not _covered_by_other(r1, c, rects, idx) for c in range(c0, c1 + 1))
    left_ok = any(not _covered_by_other(r, c0, rects, idx) for r in range(r0, r1 + 1))
    right_ok = any(not _covered_by_other(r, c1, rects, idx) for r in range(r0, r1 + 1))
    return top_ok and bottom_ok and left_ok and right_ok


def _build_rectangles(
    rng: random.Random,
    count: int,
    legend_top: bool,
) -> tuple[tuple[int, int], list[tuple[int, int, int, int]]] | None:
    cap_up = rng.randint(count, count + 3)
    cap_down = rng.randint(count, count + 3)
    cap_left = rng.randint(count, count + 4)
    cap_right = rng.randint(count, count + 4)
    pad_top = rng.randint(0, 3)
    pad_bottom = rng.randint(0, 3)
    pad_left = rng.randint(0, 3)
    pad_right = rng.randint(0, 3)

    usable_h = pad_top + cap_up + 1 + cap_down + pad_bottom
    body_h = usable_h + 1
    body_w = pad_left + cap_left + 1 + cap_right + pad_right
    if body_h < 10 or body_w < max(2 * count + 1, 10):
        return None

    if legend_top:
        center_r = 1 + pad_top + cap_up
    else:
        center_r = pad_top + cap_up
    center_c = pad_left + cap_left

    for _ in range(80):
        ups = rng.sample(range(1, cap_up + 1), count)
        downs = rng.sample(range(1, cap_down + 1), count)
        lefts = rng.sample(range(1, cap_left + 1), count)
        rights = rng.sample(range(1, cap_right + 1), count)
        rng.shuffle(ups)
        rng.shuffle(downs)
        rng.shuffle(lefts)
        rng.shuffle(rights)

        rects = []
        for i in range(count):
            rects.append(
                (
                    center_r - ups[i],
                    center_c - lefts[i],
                    center_r + downs[i],
                    center_c + rights[i],
                )
            )

        bad = False
        for i in range(count):
            for j in range(i + 1, count):
                if _contains(rects[i], rects[j]) or _contains(rects[j], rects[i]):
                    bad = True
                    break
            if bad:
                break
        if bad:
            continue

        if not all(_side_exposed(rects, i) for i in range(count)):
            continue
        return (body_h, body_w), rects
    return None


def _legend_columns(rng: random.Random, width: int, count: int) -> list[int]:
    compressed = sorted(rng.sample(range(width - count + 1), count))
    return [value + i for i, value in enumerate(compressed)]


def _strict_bg_majority(grid: np.ndarray, bg: int) -> bool:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(counts[bg]) > int(np.max(np.delete(counts, bg)))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(3, 5)
    legend_top = bool(rng.randint(0, 1))

    built = _build_rectangles(rng, count, legend_top)
    if built is None:
        return None
    body_shape, rects = built

    palette = list(range(10))
    bg = rng.choice(palette)
    palette.remove(bg)
    colors = rng.sample(palette, count)

    input_order = list(range(count))
    rng.shuffle(input_order)

    legend_order = colors[:]
    for _ in range(20):
        rng.shuffle(legend_order)
        if legend_order[-1] != colors[input_order[-1]]:
            break
    else:
        return None

    body = _render_body(body_shape, bg, rects, colors, input_order)
    if set(map(int, body[body != bg])) != set(colors):
        return None

    visible_boxes = {}
    occluded = 0
    for color, rect in zip(colors, rects):
        coords = np.argwhere(body == color)
        if coords.size == 0:
            return None
        visible_boxes[color] = _bbox(coords)
        if visible_boxes[color] != rect:
            return None
        full_area = (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1)
        if int(coords.shape[0]) < full_area:
            occluded += 1
    if occluded == 0:
        return None

    output = np.full(body_shape, bg, dtype=int)
    rect_by_color = {color: rect for color, rect in zip(colors, rects)}
    for color in legend_order:
        _paint_rect(output, rect_by_color[color], color)
    if np.array_equal(output, body):
        return None

    legend_row = np.full(body_shape[1], bg, dtype=int)
    cols = _legend_columns(rng, body_shape[1], count)
    for c, color in zip(cols, legend_order):
        legend_row[c] = color

    if legend_top:
        grid = np.vstack([legend_row, body])
    else:
        grid = np.vstack([body, legend_row])

    if not _strict_bg_majority(grid, bg):
        return None
    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to construct a valid puzzle")
