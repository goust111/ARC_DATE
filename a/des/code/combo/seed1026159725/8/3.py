import random
from collections import Counter, deque

import numpy as np


PALETTE = tuple(range(10))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 400


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _label_components(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    label = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            q = deque([(r, c)])
            labels[r, c] = label
            while q:
                cr, cc = q.popleft()
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = label
                        q.append((nr, nc))
            label += 1
    return labels


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_border(cells: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _perimeter_all_color(grid: np.ndarray, outer: tuple[int, int, int, int], color: int) -> bool:
    r0, c0, r1, c1 = outer
    return (
        np.all(grid[r0, c0 : c1 + 1] == color)
        and np.all(grid[r1, c0 : c1 + 1] == color)
        and np.all(grid[r0 : r1 + 1, c0] == color)
        and np.all(grid[r0 : r1 + 1, c1] == color)
    )


def _iter_inner(inner: tuple[int, int, int, int]):
    r0, c0, r1, c1 = inner
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            yield r, c


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _find_frames_for_color(grid: np.ndarray, color: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    frames: list[dict] = []
    for comp in _components(g != color):
        if _touches_border(comp, g.shape):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        inner_h = r1 - r0 + 1
        inner_w = c1 - c0 + 1
        if inner_h < 1 or inner_w < 1:
            continue
        if len(comp) != inner_h * inner_w:
            continue
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        outer = (r0 - 1, c0 - 1, r1 + 1, c1 + 1)
        if not _perimeter_all_color(g, outer, color):
            continue
        frames.append(
            {
                "color": int(color),
                "outer": outer,
                "inner": (r0, c0, r1, c1),
                "inner_area": int(inner_h * inner_w),
                "sample": (outer[0], outer[1]),
            }
        )
    return frames


def _infer_frames(grid: np.ndarray) -> tuple[int, int, dict, dict]:
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    candidates: list[tuple[int, int, dict, dict]] = []

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        frames = _find_frames_for_color(g, color)
        if len(frames) < 2:
            continue

        labels = _label_components(g == color)
        for frame in frames:
            sr, sc = frame["sample"]
            frame["component"] = int(labels[sr, sc])

        for i in range(len(frames)):
            for j in range(i + 1, len(frames)):
                a = frames[i]
                b = frames[j]
                if a["component"] != b["component"]:
                    continue
                if a["inner_area"] == b["inner_area"]:
                    continue
                if _boxes_overlap(a["outer"], b["outer"]):
                    continue
                score = a["inner_area"] + b["inner_area"]
                candidates.append((score, int(color), a, b))

    assert candidates, "failed to find the connected frame pair"
    _, frame_color, a, b = max(candidates, key=lambda item: item[0])
    if a["inner_area"] < b["inner_area"]:
        return bg, frame_color, a, b
    return bg, frame_color, b, a


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the smaller frame's non-background interior pixels into the larger
    frame's background slots using row-major scan order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg, _frame_color, small, large = _infer_frames(g)

    payload = [int(g[r, c]) for r, c in _iter_inner(small["inner"]) if int(g[r, c]) != bg]
    targets = [(r, c) for r, c in _iter_inner(large["inner"]) if int(g[r, c]) == bg]
    assert len(payload) <= len(targets), "small frame payload must fit in the large frame"

    out = g.copy()
    for r, c in _iter_inner(small["inner"]):
        out[r, c] = bg
    for (r, c), color in zip(targets, payload):
        out[r, c] = int(color)
    return out


def _draw_frame(grid: np.ndarray, outer: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = outer
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _paint_cells(
    grid: np.ndarray,
    inner: tuple[int, int, int, int],
    count: int,
    colors: list[int],
    rng: random.Random,
) -> list[int]:
    cells = list(_iter_inner(inner))
    rng.shuffle(cells)
    chosen = cells[:count]
    values: list[int] = []
    for r, c in chosen:
        color = int(rng.choice(colors))
        grid[r, c] = color
        values.append(color)
    return values


def _apply_expected(
    grid: np.ndarray,
    bg: int,
    small_inner: tuple[int, int, int, int],
    large_inner: tuple[int, int, int, int],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    payload = [int(grid[r, c]) for r, c in _iter_inner(small_inner) if int(grid[r, c]) != bg]
    targets = [(r, c) for r, c in _iter_inner(large_inner) if int(grid[r, c]) == bg]
    for r, c in _iter_inner(small_inner):
        out[r, c] = bg
    for (r, c), color in zip(targets, payload):
        out[r, c] = int(color)
    return out


def _gen_horizontal(rng: random.Random) -> np.ndarray | None:
    bg, frame = rng.sample(PALETTE, 2)
    fill_colors = [c for c in PALETTE if c not in {bg, frame}]
    rng.shuffle(fill_colors)
    fill_colors = fill_colors[: rng.randint(2, 5)]

    big_h = rng.randint(5, 9)
    big_w = rng.randint(6, 11)
    small_h = rng.randint(4, 7)
    small_w = rng.randint(4, 8)
    if (small_h - 2) * (small_w - 2) >= (big_h - 2) * (big_w - 2):
        return None

    gap = rng.randint(2, 8)
    top_delta = rng.randint(-(small_h - 3), big_h - 3)
    top_big = rng.randint(1, 4) + max(0, -top_delta)
    top_small = top_big + top_delta

    left_margin = rng.randint(1, 5)
    right_margin = rng.randint(1, 5)
    bottom_margin = rng.randint(1, 5)
    order_small_left = bool(rng.randint(0, 1))

    if order_small_left:
        small_left = left_margin
        big_left = small_left + small_w + gap
    else:
        big_left = left_margin
        small_left = big_left + big_w + gap

    height = max(top_big + big_h, top_small + small_h) + bottom_margin
    width = max(big_left + big_w, small_left + small_w) + right_margin
    if not (12 <= height <= 30 and 12 <= width <= 30):
        return None

    side_rows = range(max(top_big + 1, top_small + 1), min(top_big + big_h - 1, top_small + small_h - 1))
    side_rows = list(side_rows)
    if not side_rows:
        return None
    channel_r = rng.choice(side_rows)

    grid = np.full((height, width), bg, dtype=int)
    big_outer = (top_big, big_left, top_big + big_h - 1, big_left + big_w - 1)
    small_outer = (top_small, small_left, top_small + small_h - 1, small_left + small_w - 1)
    _draw_frame(grid, big_outer, frame)
    _draw_frame(grid, small_outer, frame)

    small_right = small_outer[3]
    big_left_edge = big_outer[1]
    big_right = big_outer[3]
    small_left_edge = small_outer[1]
    if order_small_left:
        grid[channel_r, small_right : big_left_edge + 1] = frame
    else:
        grid[channel_r, big_right : small_left_edge + 1] = frame

    return _finish_interior(grid, bg, frame, small_outer, big_outer, fill_colors, rng)


def _gen_vertical(rng: random.Random) -> np.ndarray | None:
    bg, frame = rng.sample(PALETTE, 2)
    fill_colors = [c for c in PALETTE if c not in {bg, frame}]
    rng.shuffle(fill_colors)
    fill_colors = fill_colors[: rng.randint(2, 5)]

    big_h = rng.randint(6, 11)
    big_w = rng.randint(5, 9)
    small_h = rng.randint(4, 8)
    small_w = rng.randint(4, 7)
    if (small_h - 2) * (small_w - 2) >= (big_h - 2) * (big_w - 2):
        return None

    gap = rng.randint(2, 8)
    left_delta = rng.randint(-(small_w - 3), big_w - 3)
    left_big = rng.randint(1, 4) + max(0, -left_delta)
    left_small = left_big + left_delta

    top_margin = rng.randint(1, 5)
    bottom_margin = rng.randint(1, 5)
    right_margin = rng.randint(1, 5)
    order_small_top = bool(rng.randint(0, 1))

    if order_small_top:
        small_top = top_margin
        big_top = small_top + small_h + gap
    else:
        big_top = top_margin
        small_top = big_top + big_h + gap

    height = max(big_top + big_h, small_top + small_h) + bottom_margin
    width = max(left_big + big_w, left_small + small_w) + right_margin
    if not (12 <= height <= 30 and 12 <= width <= 30):
        return None

    side_cols = range(max(left_big + 1, left_small + 1), min(left_big + big_w - 1, left_small + small_w - 1))
    side_cols = list(side_cols)
    if not side_cols:
        return None
    channel_c = rng.choice(side_cols)

    grid = np.full((height, width), bg, dtype=int)
    big_outer = (big_top, left_big, big_top + big_h - 1, left_big + big_w - 1)
    small_outer = (small_top, left_small, small_top + small_h - 1, left_small + small_w - 1)
    _draw_frame(grid, big_outer, frame)
    _draw_frame(grid, small_outer, frame)

    big_bottom = big_outer[2]
    small_top_edge = small_outer[0]
    small_bottom = small_outer[2]
    big_top_edge = big_outer[0]
    if order_small_top:
        grid[small_bottom : big_top_edge + 1, channel_c] = frame
    else:
        grid[big_bottom : small_top_edge + 1, channel_c] = frame

    return _finish_interior(grid, bg, frame, small_outer, big_outer, fill_colors, rng)


def _finish_interior(
    grid: np.ndarray,
    bg: int,
    frame: int,
    outer_a: tuple[int, int, int, int],
    outer_b: tuple[int, int, int, int],
    fill_colors: list[int],
    rng: random.Random,
) -> np.ndarray | None:
    inner_a = (outer_a[0] + 1, outer_a[1] + 1, outer_a[2] - 1, outer_a[3] - 1)
    inner_b = (outer_b[0] + 1, outer_b[1] + 1, outer_b[2] - 1, outer_b[3] - 1)
    area_a = (inner_a[2] - inner_a[0] + 1) * (inner_a[3] - inner_a[1] + 1)
    area_b = (inner_b[2] - inner_b[0] + 1) * (inner_b[3] - inner_b[1] + 1)
    if area_a == area_b:
        return None

    if area_a < area_b:
        small_inner, large_inner = inner_a, inner_b
    else:
        small_inner, large_inner = inner_b, inner_a

    small_area = (small_inner[2] - small_inner[0] + 1) * (small_inner[3] - small_inner[1] + 1)
    large_area = (large_inner[2] - large_inner[0] + 1) * (large_inner[3] - large_inner[1] + 1)

    large_prefilled = rng.randint(0, min(6, max(0, large_area // 3)))
    max_payload = min(small_area, large_area - large_prefilled)
    if max_payload < 1:
        return None

    if rng.random() < 0.5:
        payload_size = rng.randint(1, max(1, min(max_payload, max(1, max_payload // 2 + 1))))
    else:
        payload_size = rng.randint(max(1, max_payload // 2), max_payload)

    _paint_cells(grid, large_inner, large_prefilled, fill_colors, rng)
    _paint_cells(grid, small_inner, payload_size, fill_colors, rng)

    expected = _apply_expected(grid, bg, small_inner, large_inner)
    try:
        inferred_bg, inferred_frame, inferred_small, inferred_large = _infer_frames(grid)
        assert inferred_bg == bg
        assert inferred_frame == frame
        assert inferred_small["inner"] == small_inner
        assert inferred_large["inner"] == large_inner
        actual = main(grid)
        assert np.array_equal(actual, expected)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_horizontal(rng) if rng.random() < 0.5 else _gen_vertical(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
