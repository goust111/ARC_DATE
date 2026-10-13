from __future__ import annotations

import random

import numpy as np


BLACK = 0
ORANGE = 7
WINDOW_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)
SIDES = ("top", "bottom", "left", "right")

MIN_WINDOWS = 2
MAX_WINDOWS = 5
MIN_RECT = 3
MAX_RECT = 6
GRID_MAX = 26
GEN_RETRIES = 400
PLACEMENT_RETRIES = 120


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _boxes_overlap(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    *,
    pad: int = 0,
) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _copy_bbox(r0: int, c0: int, h: int, w: int, side: str) -> tuple[int, int, int, int]:
    if side == "top":
        return (r0 - h, c0, r0 - 1, c0 + w - 1)
    if side == "bottom":
        return (r0 + h, c0, r0 + 2 * h - 1, c0 + w - 1)
    if side == "left":
        return (r0, c0 - w, r0 + h - 1, c0 - 1)
    if side == "right":
        return (r0, c0 + w, r0 + h - 1, c0 + 2 * w - 1)
    raise ValueError(f"unknown side: {side}")


def _clip_bbox(
    box: tuple[int, int, int, int],
    h: int,
    w: int,
) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = box
    r0 = max(r0, 0)
    c0 = max(c0, 0)
    r1 = min(r1, h - 1)
    c1 = min(c1, w - 1)
    if r0 > r1 or c0 > c1:
        return None
    return (r0, c0, r1, c1)


def _find_windows(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    windows: list[dict[str, object]] = []
    for coords in _components4(g != BLACK):
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if len(coords) != (r1 - r0 + 1) * (c1 - c0 + 1):
            raise AssertionError("each foreground component must fill a rectangle")

        colors = {int(g[r, c]) for r, c in coords}
        handle_cells = [(r, c) for r, c in coords if int(g[r, c]) == ORANGE]
        if len(handle_cells) != 1:
            raise AssertionError("each window must contain exactly one orange handle")
        body_colors = colors - {ORANGE}
        if len(body_colors) != 1:
            raise AssertionError("each window must have one body color")
        body_color = next(iter(body_colors))

        hr, hc = handle_cells[0]
        side_hits = []
        if hr == r0:
            side_hits.append("top")
        if hr == r1:
            side_hits.append("bottom")
        if hc == c0:
            side_hits.append("left")
        if hc == c1:
            side_hits.append("right")
        if len(side_hits) != 1:
            raise AssertionError("handle side must be unique")

        windows.append(
            {
                "bbox": (r0, c0, r1, c1),
                "color": int(body_color),
                "side": side_hits[0],
            }
        )

    windows.sort(key=lambda item: tuple(item["bbox"]))  # deterministic overlays
    return windows


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror each rectangular window outward across the edge touched by its orange handle.

    Input structure:
    - The background is black.
    - The foreground consists of several separated solid rectangles ("windows"), each in its own color.
    - Every window contains exactly one orange handle pixel on exactly one outer row or outer column of that rectangle.

    Rule:
    - For each window, find the orange handle and determine whether it lies on the top, bottom, left, or right edge of the rectangle.
    - Reflect the entire rectangle across that outer edge and paint the reflected copy immediately outside the original window.
    - Keep the original window unchanged, but replace the reflected handle pixel with the window's body color so the copy is a solid rectangle.
    - Clip reflected pixels that fall outside the grid, and let reflected pixels overwrite any earlier content they land on.

    Key insight:
    - The orange pixel is not decoration: its edge position selects the reflection axis, and only the reflected copy loses the orange marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    out = g.copy()
    for window in _find_windows(g):
        r0, c0, r1, c1 = window["bbox"]
        color = int(window["color"])
        side = str(window["side"])
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if side == "top":
                    rr, cc = 2 * r0 - 1 - r, c
                elif side == "bottom":
                    rr, cc = 2 * r1 + 1 - r, c
                elif side == "left":
                    rr, cc = r, 2 * c0 - 1 - c
                else:
                    rr, cc = r, 2 * c1 + 1 - c
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = color if int(g[r, c]) == ORANGE else int(g[r, c])
    return out


def _sample_sides(count: int) -> list[str]:
    order = list(SIDES)
    random.shuffle(order)
    result = order[: min(count, len(order))]
    while len(result) < count:
        result.append(random.choice(SIDES))
    random.shuffle(result)
    return result


def _position_candidates(
    grid_h: int,
    grid_w: int,
    rect_h: int,
    rect_w: int,
    side: str,
    *,
    clipped: bool,
) -> list[tuple[int, int]]:
    if clipped:
        if side == "top":
            r_values = range(1, min(rect_h, grid_h - rect_h + 1))
            c_values = range(0, grid_w - rect_w + 1)
        elif side == "bottom":
            r_values = range(max(0, grid_h - 2 * rect_h + 1), grid_h - rect_h)
            c_values = range(0, grid_w - rect_w + 1)
        elif side == "left":
            r_values = range(0, grid_h - rect_h + 1)
            c_values = range(1, min(rect_w, grid_w - rect_w + 1))
        else:
            r_values = range(0, grid_h - rect_h + 1)
            c_values = range(max(0, grid_w - 2 * rect_w + 1), grid_w - rect_w)
    else:
        if side == "top":
            r_values = range(rect_h, grid_h - rect_h + 1)
            c_values = range(0, grid_w - rect_w + 1)
        elif side == "bottom":
            r_values = range(0, grid_h - 2 * rect_h + 1)
            c_values = range(0, grid_w - rect_w + 1)
        elif side == "left":
            r_values = range(0, grid_h - rect_h + 1)
            c_values = range(rect_w, grid_w - rect_w + 1)
        else:
            r_values = range(0, grid_h - rect_h + 1)
            c_values = range(0, grid_w - 2 * rect_w + 1)

    return [(r0, c0) for r0 in r_values for c0 in c_values]


def _handle_cell(r0: int, c0: int, h: int, w: int, side: str) -> tuple[int, int]:
    if side == "top":
        return (r0, c0 + random.randint(1, w - 2))
    if side == "bottom":
        return (r0 + h - 1, c0 + random.randint(1, w - 2))
    if side == "left":
        return (r0 + random.randint(1, h - 2), c0)
    if side == "right":
        return (r0 + random.randint(1, h - 2), c0 + w - 1)
    raise ValueError(f"unknown side: {side}")


def _place_window(
    grid_h: int,
    grid_w: int,
    side: str,
    color: int,
    *,
    clipped: bool,
    original_boxes: list[tuple[int, int, int, int]],
    copy_boxes: list[tuple[int, int, int, int]],
) -> dict[str, object] | None:
    for _ in range(PLACEMENT_RETRIES):
        rect_h = random.randint(MIN_RECT, MAX_RECT)
        rect_w = random.randint(MIN_RECT, MAX_RECT)
        candidates = _position_candidates(grid_h, grid_w, rect_h, rect_w, side, clipped=clipped)
        if not candidates:
            continue
        random.shuffle(candidates)
        for r0, c0 in candidates:
            bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            full_copy = _copy_bbox(r0, c0, rect_h, rect_w, side)
            visible_copy = _clip_bbox(full_copy, grid_h, grid_w)
            if visible_copy is None:
                continue
            if any(_boxes_overlap(bbox, other, pad=1) for other in original_boxes):
                continue
            if any(_boxes_overlap(bbox, other) for other in copy_boxes):
                continue
            if any(_boxes_overlap(visible_copy, other) for other in original_boxes):
                continue
            if any(_boxes_overlap(visible_copy, other) for other in copy_boxes):
                continue
            return {
                "bbox": bbox,
                "copy_box": visible_copy,
                "handle": _handle_cell(r0, c0, rect_h, rect_w, side),
                "color": color,
                "side": side,
                "clipped": clipped,
            }
    return None


def _gen_attempt() -> np.ndarray | None:
    scenario = random.choice(("plain", "clipped"))
    window_count = random.randint(MIN_WINDOWS, MAX_WINDOWS)
    min_side = 12 + 2 * window_count
    if min_side > GRID_MAX:
        return None
    grid_h = random.randint(min_side, GRID_MAX)
    grid_w = random.randint(min_side, GRID_MAX)

    colors = random.sample(WINDOW_COLORS, window_count)
    sides = _sample_sides(window_count)
    special_idx = random.randrange(window_count) if scenario == "clipped" else -1

    original_boxes: list[tuple[int, int, int, int]] = []
    copy_boxes: list[tuple[int, int, int, int]] = []
    windows: list[dict[str, object]] = []

    for idx in range(window_count):
        window = _place_window(
            grid_h,
            grid_w,
            sides[idx],
            colors[idx],
            clipped=idx == special_idx,
            original_boxes=original_boxes,
            copy_boxes=copy_boxes,
        )
        if window is None:
            return None
        original_boxes.append(window["bbox"])
        copy_boxes.append(window["copy_box"])
        windows.append(window)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for window in windows:
        r0, c0, r1, c1 = window["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(window["color"])
        hr, hc = window["handle"]
        grid[hr, hc] = ORANGE

    try:
        output = main(grid)
    except Exception:
        return None
    if np.array_equal(output, grid):
        return None
    if scenario == "clipped" and not any(bool(window["clipped"]) for window in windows):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-window input")
