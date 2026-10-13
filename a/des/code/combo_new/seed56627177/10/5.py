from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
GREY = 5
OBJECT_COLORS = tuple(color for color in range(10) if color not in (BLACK, GREY))

DIVIDER_WIDTH = 1
STRIP_COUNT_OPTIONS = (3, 5, 7)
STRIP_WIDTH_OPTIONS = (3, 5, 7)
HEIGHT_RANGE = (9, 24)
GENERATION_ATTEMPTS = 400
SHAPE_ATTEMPTS = 80

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, int(mask.shape[0])))
    return runs


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        queue = deque([(rr, cc)])
        seen[rr, cc] = True
        coords: list[tuple[int, int]] = []
        while queue:
            cr, cc = queue.popleft()
            coords.append((cr, cc))
            for dr, dc in NEIGHBORS4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        comps.append(coords)
    return comps


def _parse_layout(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    for divider_color in map(int, np.unique(g)):
        full_cols = np.all(g == divider_color, axis=0)
        divider_runs = _true_runs(full_cols)
        if len(divider_runs) < 2:
            continue
        if divider_runs[0][0] == 0 or divider_runs[-1][1] == w:
            continue

        strip_spans: list[tuple[int, int]] = []
        prev = 0
        valid = True
        for c0, c1 in divider_runs:
            if c0 <= prev:
                valid = False
                break
            strip_spans.append((prev, c0))
            prev = c1
        if not valid or prev >= w:
            continue
        strip_spans.append((prev, w))

        strip_widths = {c1 - c0 for c0, c1 in strip_spans}
        if len(strip_widths) != 1:
            continue
        strip_width = strip_widths.pop()
        if strip_width < 3 or strip_width % 2 == 0:
            continue
        if len(strip_spans) < 3 or len(strip_spans) % 2 == 0:
            continue

        non_divider = np.ones((h, w), dtype=bool)
        for c0, c1 in divider_runs:
            non_divider[:, c0:c1] = False
        values = g[non_divider]
        if values.size == 0:
            continue
        bg = int(Counter(map(int, values)).most_common(1)[0][0])

        odd_objects: list[dict] = []
        even_seeds: list[dict] = []
        valid = True
        for strip_index, (c0, c1) in enumerate(strip_spans):
            strip = g[:, c0:c1]
            if np.any(strip == divider_color):
                valid = False
                break
            fg_mask = strip != bg
            if strip_index % 2 == 0:
                colors = set(map(int, strip[fg_mask]))
                if len(colors) != 1:
                    valid = False
                    break
                color = int(next(iter(colors)))
                obj_mask = strip == color
                comps = _components(obj_mask)
                if len(comps) != 1:
                    valid = False
                    break
                rows = np.argwhere(obj_mask)[:, 0]
                odd_objects.append(
                    {
                        "strip_index": strip_index,
                        "color": color,
                        "height": int(rows.max() - rows.min() + 1),
                        "span": (c0, c1),
                    }
                )
            else:
                coords = np.argwhere(fg_mask)
                if coords.shape[0] != 1:
                    valid = False
                    break
                rr, cc = coords[0]
                center = strip_width // 2
                if int(cc) != center:
                    valid = False
                    break
                even_seeds.append(
                    {
                        "strip_index": strip_index,
                        "row": int(rr),
                        "col": int(c0 + center),
                        "color": int(strip[rr, cc]),
                    }
                )

        if not valid:
            continue
        odd_colors = [obj["color"] for obj in odd_objects]
        if len(set(odd_colors)) != len(odd_colors):
            continue

        for seed in even_seeds:
            left_obj = odd_objects[(seed["strip_index"] - 1) // 2]
            right_obj = odd_objects[(seed["strip_index"] + 1) // 2]
            if seed["color"] == left_obj["color"]:
                matched = left_obj
            elif seed["color"] == right_obj["color"]:
                matched = right_obj
            else:
                valid = False
                break
            half = matched["height"] // 2
            if seed["row"] - half < 0 or seed["row"] + half >= h:
                valid = False
                break
            seed["matched_height"] = matched["height"]

        if not valid:
            continue

        return {
            "bg": bg,
            "divider_color": divider_color,
            "strip_width": strip_width,
            "strip_spans": strip_spans,
            "odd_objects": odd_objects,
            "even_seeds": even_seeds,
        }

    raise AssertionError("could not parse vertical strip layout")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow three-column symmetric bars in seed strips using the height of a same-color neighboring object.

    Input structure:
    - The background is one color, and full-height divider columns of another color split
      the grid into equal-width vertical strips of odd width.
    - Odd-numbered strips each contain exactly one 4-connected monochromatic object, and
      these object colors are all distinct.
    - Even-numbered strips each contain a single seed pixel on the strip's center column;
      its color matches the object in exactly one adjacent odd strip.

    Rule:
    - Parse the divider columns to recover the strip boundaries and identify the background.
    - For each even strip, find the adjacent odd-strip object with the same color as the seed
      and measure that object's height as bbox-row-span.
    - Let half-span = floor(height / 2). Centered on the seed row, fill a vertical segment
      from row-half-span to row+half-span on the center column and on the columns one step
      left and right of center.
    - Leave the divider columns and all odd-strip objects unchanged.

    Key insight:
    - The seed pixel is an instruction token: its color selects which neighboring object to
      read, and that object's vertical extent determines how far the centered 3-column bar grows.
    """
    g = np.asarray(grid, dtype=int)
    layout = _parse_layout(g)
    out = g.copy()

    for seed in layout["even_seeds"]:
        half = int(seed["matched_height"]) // 2
        r0 = int(seed["row"]) - half
        r1 = int(seed["row"]) + half
        c = int(seed["col"])
        out[r0 : r1 + 1, c - 1 : c + 2] = int(seed["color"])

    return out


def _sample_strip_params() -> tuple[int, int, int]:
    feasible: list[tuple[int, int]] = []
    for strip_count in STRIP_COUNT_OPTIONS:
        for strip_width in STRIP_WIDTH_OPTIONS:
            total_width = strip_count * strip_width + (strip_count - 1) * DIVIDER_WIDTH
            if total_width <= 30:
                feasible.append((strip_count, strip_width))
    strip_count, strip_width = random.choice(feasible)
    height = random.randint(*HEIGHT_RANGE)
    return height, strip_count, strip_width


def _choose_with_parity(values: list[int]) -> int:
    parity = random.randrange(2)
    matched = [value for value in values if value % 2 == parity]
    return random.choice(matched or values)


def _random_row_intervals(height: int, width: int) -> list[tuple[int, int]] | None:
    for _ in range(SHAPE_ATTEMPTS):
        intervals: list[tuple[int, int]] = []
        prev: tuple[int, int] | None = None
        for _row in range(height):
            candidates: list[tuple[int, int]] = []
            for seg_width in range(1, width + 1):
                for left in range(0, width - seg_width + 1):
                    right = left + seg_width - 1
                    if prev is None or max(left, prev[0]) <= min(right, prev[1]):
                        candidates.append((left, right))
            if not candidates:
                break
            if prev is None:
                choices = [span for span in candidates if span[1] - span[0] + 1 < width]
                interval = random.choice(choices or candidates)
            else:
                interval = random.choice(candidates)
            intervals.append(interval)
            prev = interval
        if len(intervals) != height:
            continue
        widths = [right - left + 1 for left, right in intervals]
        if height > 1 and width > 1:
            if all(seg_width == width for seg_width in widths):
                continue
            if len(set(intervals)) == 1 and widths[0] == 1:
                continue
        return intervals
    return None


def _make_object_patch(height: int, width: int) -> np.ndarray | None:
    intervals = _random_row_intervals(height, width)
    if intervals is None:
        return None
    patch = np.zeros((height, width), dtype=bool)
    for row, (left, right) in enumerate(intervals):
        patch[row, left : right + 1] = True
    return patch


def _place_object_in_strip(
    grid: np.ndarray,
    strip_left: int,
    strip_width: int,
    color: int,
) -> dict | None:
    h = int(grid.shape[0])
    valid_heights = list(range(1, h - 1))
    obj_height = _choose_with_parity(valid_heights)

    max_obj_width = min(strip_width, 4 if strip_width >= 4 else strip_width)
    min_obj_width = 2 if max_obj_width >= 2 and random.random() < 0.8 else 1
    obj_width = random.randint(min_obj_width, max_obj_width)

    patch = _make_object_patch(obj_height, obj_width)
    if patch is None:
        return None

    vertical_slack = h - obj_height
    if vertical_slack >= 2:
        top = random.randint(1, vertical_slack - 1)
    else:
        top = random.randint(0, vertical_slack)

    horizontal_slack = strip_width - obj_width
    if horizontal_slack >= 2:
        left = random.randint(1, horizontal_slack - 1)
    else:
        left = random.randint(0, horizontal_slack)

    rows, cols = np.where(patch)
    grid[top + rows, strip_left + left + cols] = color
    return {"color": color, "height": obj_height}


def _build_input() -> np.ndarray | None:
    height, strip_count, strip_width = _sample_strip_params()
    width = strip_count * strip_width + (strip_count - 1) * DIVIDER_WIDTH
    grid = np.full((height, width), BLACK, dtype=int)

    strip_spans: list[tuple[int, int]] = []
    col = 0
    for strip_index in range(strip_count):
        strip_spans.append((col, col + strip_width))
        col += strip_width
        if strip_index < strip_count - 1:
            grid[:, col : col + DIVIDER_WIDTH] = GREY
            col += DIVIDER_WIDTH

    odd_count = (strip_count + 1) // 2
    colors = random.sample(OBJECT_COLORS, odd_count)
    odd_objects: list[dict] = []
    for obj_index, color in enumerate(colors):
        strip_left, _strip_right = strip_spans[2 * obj_index]
        obj = _place_object_in_strip(grid, strip_left, strip_width, color)
        if obj is None:
            return None
        odd_objects.append(obj)

    for even_strip_index in range(1, strip_count, 2):
        left_obj = odd_objects[(even_strip_index - 1) // 2]
        right_obj = odd_objects[(even_strip_index + 1) // 2]
        chosen = left_obj if random.random() < 0.5 else right_obj
        half = chosen["height"] // 2
        seed_row = random.randint(half, height - 1 - half)
        strip_left, strip_right = strip_spans[even_strip_index]
        center_col = (strip_left + strip_right - 1) // 2
        grid[seed_row, center_col] = chosen["color"]

    try:
        layout = _parse_layout(grid)
        assert len(layout["strip_spans"]) == strip_count
        assert len(layout["odd_objects"]) == odd_count
        assert len(layout["even_seeds"]) == (strip_count - 1) // 2
        _ = main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid strip puzzle")
