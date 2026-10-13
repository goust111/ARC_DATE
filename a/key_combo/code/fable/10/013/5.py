from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return comps


def component_signature(component: dict) -> dict:
    r0, c0, r1, c1 = component["bbox"]
    local = {(r - r0, c - c0) for r, c in component["coords"]}
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    def degree(cell: Coord) -> int:
        rr, cc = cell
        return sum((nr, nc) in local for nr, nc in neighbors4(rr, cc))

    degrees = {cell: degree(cell) for cell in local}
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "each fragment must be a simple path"
    assert all(deg in (1, 2) for deg in degrees.values()), "path cells must have degree 1 or 2"

    min_c = min(c for _r, c in local)
    max_c = max(c for _r, c in local)
    left = [r for r, c in endpoints if c == min_c]
    right = [r for r, c in endpoints if c == max_c]
    assert len(left) == 1 and len(right) == 1, "outer columns must each contain one endpoint"

    return {
        "color": int(component["color"]),
        "area": int(component["area"]),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "coords_local": sorted(local),
        "height": int(h),
        "width": int(w),
        "left_row": int(left[0]),
        "right_row": int(right[0]),
    }


def _render_sorted_chain(segments: list[dict], height: int, bg: int) -> tuple[np.ndarray, list[int]]:
    out_w = sum(seg["width"] for seg in segments)
    out = np.full((height, out_w), bg, dtype=int)
    x = 0
    shifts: list[int] = []
    prev_right_abs = None

    for idx, seg in enumerate(segments):
        if idx == 0:
            top = int(seg["bbox"][0])
        else:
            assert prev_right_abs is not None
            top = int(prev_right_abs - seg["left_row"])
        assert 0 <= top <= height - seg["height"]
        shifts.append(top - int(seg["bbox"][0]))

        for rr, cc in seg["coords_local"]:
            out[top + rr, x + cc] = seg["color"]
        prev_right_abs = top + seg["right_row"]
        x += seg["width"]

    return out, shifts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder scattered monochrome path fragments by area and concatenate them into one endpoint-linked chain.

    Input structure:
    - The background is the most common color.
    - Several disconnected monochrome path fragments are scattered in the grid.
    - In every fragment, the leftmost occupied column and the rightmost occupied column each contain exactly one degree-1 endpoint pixel.
    - Fragment areas are pairwise distinct.

    Rule:
    - Sort the fragments by pixel count from largest to smallest.
    - Keep the largest fragment at its original row position and move it to the left edge of the output.
    - For each later fragment, apply only a vertical translation so its left endpoint is on the same row as the previous placed fragment's right endpoint, then place it immediately to the right.
    - The output keeps the input height and uses the sum of fragment widths as its width.

    Key insight:
    - Input positions only locate the pieces; the true placement order comes from area, while the endpoint rows determine the required vertical offsets during concatenation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0

    bg = get_bg(g)
    comps = find_components(g, bg)
    assert len(comps) >= 2

    segments = [component_signature(comp) for comp in comps]
    areas = [seg["area"] for seg in segments]
    assert len(set(areas)) == len(areas), "fragment areas must define a unique order"

    ordered = sorted(segments, key=lambda seg: seg["area"], reverse=True)
    out, _shifts = _render_sorted_chain(ordered, g.shape[0], bg)
    return out


SEGMENT_COUNT_RANGE = (3, 5)
GRID_HEIGHT_RANGE = (8, 12)
LOCAL_HEIGHT_RANGE = (2, 4)
SEGMENT_WIDTH_RANGE = (2, 6)
INPUT_GAP_RANGE = (2, 4)
MAX_TRIES = 400


def _build_path_from_rows(rows: list[int]) -> list[Coord]:
    path = [(int(rows[0]), 0)]
    cur = int(rows[0])
    for c in range(1, len(rows)):
        target = int(rows[c])
        step = 1 if target > cur else -1
        while cur != target:
            cur += step
            path.append((cur, c - 1))
        path.append((cur, c))
    return path


def _normalize_path(path: list[Coord], color: int) -> dict:
    min_r = min(r for r, _ in path)
    min_c = min(c for _, c in path)
    coords = sorted({(r - min_r, c - min_c) for r, c in path})
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return {
        "color": int(color),
        "coords_local": coords,
        "height": max(rows) - min(rows) + 1,
        "width": max(cols) - min(cols) + 1,
        "area": len(coords),
        "left_row": int(next(r for r, c in coords if c == 0)),
        "bbox": None,
    }


def _right_endpoint_row(coords_local: list[Coord], width: int) -> int:
    local = set(coords_local)

    def degree(cell: Coord) -> int:
        rr, cc = cell
        return sum((nr, nc) in local for nr, nc in neighbors4(rr, cc))

    rows = [r for r, c in coords_local if c == width - 1 and degree((r, c)) == 1]
    assert len(rows) == 1
    return int(rows[0])


def _random_fragment(
    color: int,
    area_forbidden: set[int],
    connector_row: int | None,
    height: int,
    *,
    max_area: int | None = None,
    min_area: int = 0,
) -> dict | None:
    for _ in range(120):
        local_h = random.randint(*LOCAL_HEIGHT_RANGE)
        width = random.randint(*SEGMENT_WIDTH_RANGE)
        rows = [random.randrange(local_h)]
        for _c in range(1, width):
            rows.append(random.randrange(local_h))
        if all(rows[c] == rows[c - 1] for c in range(1, width)) and random.random() < 0.7:
            continue

        path = _build_path_from_rows(rows)
        frag = _normalize_path(path, color)
        frag["left_row"] = int(rows[0] - min(rows))
        frag["right_row"] = _right_endpoint_row(frag["coords_local"], frag["width"])
        if frag["area"] in area_forbidden:
            continue
        if max_area is not None and frag["area"] >= max_area:
            continue
        if frag["area"] < min_area:
            continue
        if connector_row is not None and not (0 <= connector_row - frag["left_row"] <= height - frag["height"]):
            continue
        local = set(frag["coords_local"])
        degrees = [sum((nr, nc) in local for nr, nc in neighbors4(r, c)) for r, c in local]
        if degrees.count(1) != 2 or any(deg not in (1, 2) for deg in degrees):
            continue
        return frag
    return None


def _choose_palette(n: int) -> tuple[int, list[int]]:
    bg = random.randint(0, 9)
    pool = [c for c in range(10) if c != bg]
    base = random.sample(pool, k=min(len(pool), max(2, n - 1)))
    colors = [random.choice(base) for _ in range(n)]
    random.shuffle(colors)
    return bg, colors


def _build_output_segments(height: int, colors: list[int]) -> list[dict] | None:
    used_areas: set[int] = set()
    segments: list[dict] = []
    prev_right_abs = None
    prev_area = None

    for idx, color in enumerate(colors):
        connector = None if idx == 0 else int(prev_right_abs)
        min_area = len(colors) - idx + 2
        frag = _random_fragment(color, used_areas, connector, height, max_area=prev_area, min_area=min_area)
        if frag is None:
            return None

        if idx == 0:
            top_choices = [r for r in range(0, height - frag["height"] + 1)]
            top_choices = [r for r in top_choices if 1 <= r + frag["right_row"] <= height - 2] or top_choices
            top = random.choice(top_choices)
        else:
            top = int(prev_right_abs - frag["left_row"])

        frag = dict(frag)
        frag["bbox"] = (int(top), 0, int(top + frag["height"] - 1), int(frag["width"] - 1))
        prev_right_abs = int(top + frag["right_row"])
        prev_area = int(frag["area"])
        used_areas.add(int(frag["area"]))
        segments.append(frag)

    areas = [seg["area"] for seg in segments]
    if len(set(areas)) != len(areas) or areas != sorted(areas, reverse=True):
        return None
    return segments


def _place_input_segments(height: int, ordered: list[dict], bg: int) -> np.ndarray | None:
    n = len(ordered)
    widths = [seg["width"] for seg in ordered]
    gaps = [random.randint(*INPUT_GAP_RANGE) for _ in range(n - 1)]
    total_w = sum(widths) + sum(gaps)
    if total_w > 30:
        return None

    perm = list(range(n))
    while perm == list(range(n)):
        random.shuffle(perm)

    grid = np.full((height, total_w), bg, dtype=int)
    x = 0
    x_positions = [0] * n
    for slot, seg_idx in enumerate(perm):
        x_positions[seg_idx] = x
        x += ordered[seg_idx]["width"]
        if slot < n - 1:
            x += gaps[slot]

    for seg_idx, seg in enumerate(ordered):
        if seg_idx == 0:
            top = int(seg["bbox"][0])
        else:
            top = random.randint(0, height - seg["height"])
        for rr, cc in seg["coords_local"]:
            grid[top + rr, x_positions[seg_idx] + cc] = seg["color"]
        seg["bbox"] = (
            int(top),
            int(x_positions[seg_idx]),
            int(top + seg["height"] - 1),
            int(x_positions[seg_idx] + seg["width"] - 1),
        )

    x_order = sorted(range(n), key=lambda i: ordered[i]["bbox"][1])
    if x_order == list(range(n)):
        return None
    return grid


def _expected_from_input(grid: np.ndarray) -> tuple[np.ndarray, list[int], list[dict]]:
    bg = get_bg(grid)
    comps = find_components(grid, bg)
    segs = [component_signature(comp) for comp in comps]
    ordered = sorted(segs, key=lambda seg: seg["area"], reverse=True)
    out, shifts = _render_sorted_chain(ordered, grid.shape[0], bg)
    return out, shifts, ordered


def _validate_case(inp: np.ndarray) -> bool:
    bg = get_bg(inp)
    comps = find_components(inp, bg)
    if not (3 <= len(comps) <= 5):
        return False

    segs = [component_signature(comp) for comp in comps]
    if len({seg["area"] for seg in segs}) != len(segs):
        return False
    if sum(seg["width"] for seg in segs) > 30:
        return False

    x_order = sorted(range(len(segs)), key=lambda i: segs[i]["bbox"][1])
    area_order = sorted(range(len(segs)), key=lambda i: segs[i]["area"], reverse=True)
    if x_order == area_order:
        return False

    out, shifts, ordered = _expected_from_input(inp)
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if not any(shift != 0 for shift in shifts[1:]):
        return False
    if not any(seg["height"] > 1 for seg in ordered):
        return False

    solved = main(inp)
    if not np.array_equal(solved, out):
        return False

    out_comps = find_components(out, bg)
    if len(out_comps) != 1:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    n = random.randint(*SEGMENT_COUNT_RANGE)
    bg, colors = _choose_palette(n)

    ordered = _build_output_segments(height, colors)
    if ordered is None:
        return None

    output_width = sum(seg["width"] for seg in ordered)
    if output_width > 30:
        return None

    inp = _place_input_segments(height, [dict(seg) for seg in ordered], bg)
    if inp is None:
        return None
    counts = Counter(map(int, inp.ravel()))
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return None
    if not _validate_case(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_TRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid area-sorted endpoint-chain input")
