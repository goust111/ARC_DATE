from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
DIVIDER = 5
MARKER = 9
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8)

HEIGHT_MIN = 10
HEIGHT_MAX = 18
LEFT_WIDTH_MIN = 11
LEFT_WIDTH_MAX = 22
OBJECT_COUNT_MIN = 3
OBJECT_COUNT_MAX = 5
RECT_HEIGHT_MIN = 2
RECT_HEIGHT_MAX = 6
RECT_WIDTH_MIN = 2
RECT_WIDTH_MAX = 4
MAX_ROW_OVERLAP = 3
MAX_GENERATION_ATTEMPTS = 400

RNG = np.random.default_rng()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))

    return comps


def _extract_objects(grid: np.ndarray) -> tuple[list[dict[str, int | bool]], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 3, "expected left region, divider, and projection column"

    bg = int(g[0, -1])
    assert np.all(g[:, -1] == bg), "right projection column must start as a solid background strip"

    divider = int(g[0, -2])
    assert divider != bg, "divider must differ from the background"
    assert np.all(g[:, -2] == divider), "expected a full-height divider column"

    left = g[:, :-2]
    mask = left != bg
    comps = _components(mask)
    assert comps, "expected at least one source rectangle"

    objects: list[dict[str, int | bool]] = []
    for coords in comps:
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        assert c0 != c1, "rectangle width must be at least 2 so the edge marker has a side"

        patch = left[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch != bg), "each object bbox must be a solid rectangle"

        colors = {int(v) for v in np.unique(patch)}
        assert MARKER in colors, "each rectangle must contain one maroon marker"
        body_colors = [color for color in colors if color != MARKER]
        assert len(body_colors) == 1, "each rectangle must otherwise be monochrome"
        body = int(body_colors[0])

        marker_pts = np.argwhere(patch == MARKER)
        assert len(marker_pts) == 1, "each rectangle must have exactly one marker pixel"
        mr, mc = map(int, marker_pts[0])
        mc += c0

        if mc == c0:
            front = True
        else:
            assert mc == c1, "marker must sit on the left or right edge"
            front = False

        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        objects.append(
            {
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
                "area": area,
                "color": body,
                "front": front,
            }
        )

    return objects, bg


def _row_winner(objects: list[dict[str, int | bool]], row: int) -> int | None:
    active = [obj for obj in objects if int(obj["r0"]) <= row <= int(obj["r1"])]
    if not active:
        return None

    front_active = [obj for obj in active if bool(obj["front"])]
    candidates = front_active if front_active else active

    best = min(
        candidates,
        key=lambda obj: (
            int(obj["area"]),
            int(obj["r0"]),
            int(obj["c0"]),
            int(obj["color"]),
        ),
    )
    return int(best["color"])


def main(grid: np.ndarray) -> np.ndarray:
    """Project left-side rectangles into the one-column strip, resolving row overlap by marker depth."""
    g = np.asarray(grid, dtype=int)
    objects, bg = _extract_objects(g)

    out = np.full((g.shape[0], 1), bg, dtype=int)
    for row in range(g.shape[0]):
        winner = _row_winner(objects, row)
        if winner is not None:
            out[row, 0] = winner
    return out


def _sample_columns(widths: list[int], left_width: int) -> list[int] | None:
    min_span = 2 + sum(widths) + (len(widths) - 1)
    if min_span > left_width:
        return None

    slack = left_width - min_span
    extras = np.zeros(len(widths) + 1, dtype=int)
    for _ in range(slack):
        extras[int(RNG.integers(0, len(extras)))] += 1

    cols: list[int] = []
    cur = 1 + int(extras[0])
    for i, width in enumerate(widths):
        cols.append(cur)
        cur += width
        if i != len(widths) - 1:
            cur += 1 + int(extras[i + 1])
    return cols


def _sample_covering_top(height: int, grid_h: int, band0: int, band1: int) -> int | None:
    low = max(0, band1 - height + 1)
    high = min(band0, grid_h - height)
    if low > high:
        return None
    return int(RNG.integers(low, high + 1))


def _sample_outside_band_top(height: int, grid_h: int, band0: int, band1: int) -> int | None:
    candidates = [
        top
        for top in range(grid_h - height + 1)
        if top + height - 1 < band0 or top > band1
    ]
    if not candidates:
        return None
    return int(candidates[int(RNG.integers(0, len(candidates)))])


def _covers_band(top: int, height: int, band0: int, band1: int) -> bool:
    return top <= band0 and top + height - 1 >= band1


def _build_grid(specs: list[dict[str, int | bool]], grid_h: int, left_width: int) -> np.ndarray:
    grid = np.full((grid_h, left_width + 2), BG, dtype=int)
    grid[:, left_width] = DIVIDER

    for spec in specs:
        r0 = int(spec["top"])
        c0 = int(spec["left"])
        h = int(spec["height"])
        w = int(spec["width"])
        color = int(spec["color"])
        grid[r0 : r0 + h, c0 : c0 + w] = color

        marker_row = r0 + int(RNG.integers(0, h))
        marker_col = c0 if bool(spec["front"]) else c0 + w - 1
        grid[marker_row, marker_col] = MARKER

    return grid


def _gen_attempt() -> np.ndarray | None:
    n_objects = int(RNG.integers(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX + 1))
    pattern = "double_front" if int(RNG.integers(0, 2)) == 0 else "front_back"
    special_count = 3 if pattern == "double_front" else 2

    widths = [int(RNG.integers(RECT_WIDTH_MIN, RECT_WIDTH_MAX + 1)) for _ in range(n_objects)]
    heights = [int(RNG.integers(RECT_HEIGHT_MIN, RECT_HEIGHT_MAX + 1)) for _ in range(n_objects)]

    if pattern == "double_front" and widths[0] * heights[0] == widths[1] * heights[1]:
        return None

    min_left = 2 + sum(widths) + (n_objects - 1)
    if min_left > LEFT_WIDTH_MAX:
        return None
    left_width = int(RNG.integers(max(LEFT_WIDTH_MIN, min_left), LEFT_WIDTH_MAX + 1))

    min_h = max(HEIGHT_MIN, max(heights) + 1)
    if min_h > HEIGHT_MAX:
        return None
    grid_h = int(RNG.integers(min_h, HEIGHT_MAX + 1))

    max_band = min(3, min(heights[:special_count]))
    band_len = int(RNG.integers(1, max_band + 1))
    if grid_h - band_len < 2:
        return None
    band0 = int(RNG.integers(1, grid_h - band_len))
    band1 = band0 + band_len - 1

    fronts = [True, False] if pattern == "front_back" else [True, True, False]
    while len(fronts) < n_objects:
        fronts.append(bool(RNG.integers(0, 2)))

    tops: list[int] = []
    for i in range(special_count):
        top = _sample_covering_top(heights[i], grid_h, band0, band1)
        if top is None:
            return None
        tops.append(top)
    for i in range(special_count, n_objects):
        top = _sample_outside_band_top(heights[i], grid_h, band0, band1)
        if top is None:
            return None
        tops.append(top)

    cover = np.zeros(grid_h, dtype=int)
    front_cover = np.zeros(grid_h, dtype=int)
    for top, height, front in zip(tops, heights, fronts):
        cover[top : top + height] += 1
        if front:
            front_cover[top : top + height] += 1

    if int(cover.max()) > MAX_ROW_OVERLAP:
        return None
    if not np.any(cover == 0):
        return None

    if pattern == "double_front":
        if np.any(front_cover[band0 : band1 + 1] < 2):
            return None
    else:
        if np.any(front_cover[band0 : band1 + 1] != 1):
            return None
        if np.any(cover[band0 : band1 + 1] - front_cover[band0 : band1 + 1] < 1):
            return None

    colors = list(RNG.choice(np.array(OBJECT_COLORS, dtype=int), size=n_objects, replace=False))
    specs = [
        {
            "height": heights[i],
            "width": widths[i],
            "top": tops[i],
            "front": fronts[i],
            "color": int(colors[i]),
            "special": i < special_count,
        }
        for i in range(n_objects)
    ]

    perm = list(map(int, RNG.permutation(n_objects)))
    specs = [specs[i] for i in perm]
    cols = _sample_columns([int(spec["width"]) for spec in specs], left_width)
    if cols is None:
        return None
    for spec, col in zip(specs, cols):
        spec["left"] = col

    grid = _build_grid(specs, grid_h, left_width)

    try:
        objects, _bg = _extract_objects(grid)
    except AssertionError:
        return None

    if len(objects) != n_objects:
        return None

    special_objects = [
        obj
        for obj in objects
        if _covers_band(int(obj["r0"]), int(obj["r1"]) - int(obj["r0"]) + 1, band0, band1)
    ]
    if len(special_objects) < special_count:
        return None

    front_special = [obj for obj in special_objects if bool(obj["front"])]
    back_special = [obj for obj in special_objects if not bool(obj["front"])]
    if pattern == "double_front":
        if len(front_special) < 2 or len(back_special) < 1:
            return None
        front_areas = {int(obj["area"]) for obj in front_special}
        if len(front_areas) < 2:
            return None
    else:
        if len(front_special) < 1 or len(back_special) < 1:
            return None

    out = main(grid)
    if out.shape != (grid_h, 1):
        return None
    if len(np.unique(out)) < 3:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
