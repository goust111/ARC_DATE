from __future__ import annotations

import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRECTIONS = ("up", "down", "left", "right")
GEN_RETRIES = 400
PLACE_RETRIES = 200


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _mode(arr: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _shape_and_markers(g: np.ndarray, pts: list[tuple[int, int]]) -> tuple[int, list[tuple[int, int]]]:
    vals = np.array([int(g[r, c]) for r, c in pts], dtype=int)
    colors, counts = np.unique(vals, return_counts=True)
    shape_color = int(colors[counts.argmax()])
    markers = [(r, c) for r, c in pts if int(g[r, c]) != shape_color]
    return shape_color, markers


def _direction_from_markers(
    marker_pts: list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
) -> str:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    mean_r = sum(r for r, _ in marker_pts) / len(marker_pts)
    mean_c = sum(c for _, c in marker_pts) / len(marker_pts)
    dr = mean_r - center_r
    dc = mean_c - center_c
    if abs(dc) > abs(dr):
        return "right" if dc > 0 else "left"
    return "down" if dr > 0 else "up"


def _fill_extension(
    out: np.ndarray,
    bg: int,
    color: int,
    bbox: tuple[int, int, int, int],
    direction: str,
    length: int,
) -> None:
    h, w = out.shape
    r0, c0, r1, c1 = bbox
    if direction == "up":
        rr0, rr1, cc0, cc1 = r0 - length, r0 - 1, c0, c1
    elif direction == "down":
        rr0, rr1, cc0, cc1 = r1 + 1, r1 + length, c0, c1
    elif direction == "left":
        rr0, rr1, cc0, cc1 = r0, r1, c0 - length, c0 - 1
    else:
        rr0, rr1, cc0, cc1 = r0, r1, c1 + 1, c1 + length

    rr0 = max(0, rr0)
    rr1 = min(h - 1, rr1)
    cc0 = max(0, cc0)
    cc1 = min(w - 1, cc1)
    if rr0 > rr1 or cc0 > cc1:
        return
    for r in range(rr0, rr1 + 1):
        for c in range(cc0, cc1 + 1):
            if int(out[r, c]) == bg:
                out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore marker-bearing shapes, then grow same-colored rectangles outward from the marker-biased side.

    Input structure:
    - The background is the dominant color.
    - Several separated solid foreground shapes appear as 4-connected non-background components.
    - Some shapes contain a few marker-color pixels embedded inside an otherwise single-color shape.
    - Each marked shape is still one 4-connected component when marker pixels are included.

    Rule:
    - For each non-background component that contains marker pixels, treat the majority color as the shape color and the minority pixels as markers.
    - Count the markers, repaint them back to the shape color, and compute the component bbox.
    - Compare the marker centroid with the bbox center; whichever axis has the larger offset chooses the outward side (up, down, left, or right).
    - Extend a solid rectangle from that bbox side by N cells, where N is the marker count, using the shape color and only covering background cells.

    Key insight:
    The embedded marker pixels encode two things at once: their count gives the extension length, and their positional bias inside the shape selects which bbox side grows outward.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode(g)
    out = g.copy()

    for pts in _components(g != bg):
        shape_color, markers = _shape_and_markers(g, pts)
        if not markers:
            continue
        for r, c in markers:
            out[r, c] = shape_color
        bbox = _bbox(pts)
        direction = _direction_from_markers(markers, bbox)
        _fill_extension(out, bg, shape_color, bbox, direction, len(markers))

    return out


def _centered_segment_starts(span: int, length: int) -> list[int]:
    center = (span - 1) / 2.0
    starts = list(range(span - length + 1))
    scores = [abs((start + (length - 1) / 2.0) - center) for start in starts]
    best = min(scores)
    return [start for start, score in zip(starts, scores, strict=False) if score == best]


def _marker_positions(h: int, w: int, direction: str, count: int, rng: random.Random) -> list[tuple[int, int]]:
    if direction in ("up", "down"):
        span = w - 2
        if span < count:
            raise ValueError("shape too narrow for marker segment")
        start = rng.choice(_centered_segment_starts(span, count))
        row = 1 if direction == "up" else h - 2
        return [(row, 1 + start + offset) for offset in range(count)]

    span = h - 2
    if span < count:
        raise ValueError("shape too short for marker segment")
    start = rng.choice(_centered_segment_starts(span, count))
    col = 1 if direction == "left" else w - 2
    return [(1 + start + offset, col) for offset in range(count)]


def _combined_bbox(r0: int, c0: int, h: int, w: int, direction: str, length: int) -> tuple[int, int, int, int]:
    if direction == "up":
        return r0 - length, c0, r0 + h - 1, c0 + w - 1
    if direction == "down":
        return r0, c0, r0 + h - 1 + length, c0 + w - 1
    if direction == "left":
        return r0, c0 - length, r0 + h - 1, c0 + w - 1
    return r0, c0, r0 + h - 1, c0 + w - 1 + length


def _reserve_bbox(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0 - 1, c0 - 1, r1 + 1, c1 + 1


def _fits(blocked: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return not bool(blocked[r0 : r1 + 1, c0 : c1 + 1].any())


def _mark(blocked: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _place_marked_object(
    blocked: np.ndarray,
    h_grid: int,
    w_grid: int,
    color: int,
    marker_color: int,
    rng: random.Random,
) -> dict | None:
    direction = rng.choice(DIRECTIONS)
    count = rng.randint(1, 4)
    if direction in ("up", "down"):
        h = rng.randint(4, 8)
        w = rng.randint(max(4, count + 2), 8)
    else:
        h = rng.randint(max(4, count + 2), 8)
        w = rng.randint(4, 8)

    if direction == "up":
        r_min, r_max = count + 1, h_grid - h - 2
        c_min, c_max = 1, w_grid - w - 2
    elif direction == "down":
        r_min, r_max = 1, h_grid - h - count - 2
        c_min, c_max = 1, w_grid - w - 2
    elif direction == "left":
        r_min, r_max = 1, h_grid - h - 2
        c_min, c_max = count + 1, w_grid - w - 2
    else:
        r_min, r_max = 1, h_grid - h - 2
        c_min, c_max = 1, w_grid - w - count - 2

    if r_min > r_max or c_min > c_max:
        return None

    for _ in range(PLACE_RETRIES):
        r0 = rng.randint(r_min, r_max)
        c0 = rng.randint(c_min, c_max)
        combined = _combined_bbox(r0, c0, h, w, direction, count)
        reserve = _reserve_bbox(combined)
        if not _fits(blocked, reserve):
            continue
        markers = _marker_positions(h, w, direction, count, rng)
        bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        actual = _direction_from_markers([(r0 + r, c0 + c) for r, c in markers], bbox)
        if actual != direction:
            continue
        _mark(blocked, reserve)
        return {
            "r0": r0,
            "c0": c0,
            "h": h,
            "w": w,
            "color": color,
            "marker_color": marker_color,
            "direction": direction,
            "count": count,
            "markers": markers,
        }
    return None


def _place_distractor(
    blocked: np.ndarray,
    h_grid: int,
    w_grid: int,
    color: int,
    rng: random.Random,
) -> dict | None:
    h = rng.randint(3, 7)
    w = rng.randint(3, 7)
    r_min, r_max = 1, h_grid - h - 2
    c_min, c_max = 1, w_grid - w - 2
    if r_min > r_max or c_min > c_max:
        return None

    for _ in range(PLACE_RETRIES):
        r0 = rng.randint(r_min, r_max)
        c0 = rng.randint(c_min, c_max)
        reserve = _reserve_bbox((r0, c0, r0 + h - 1, c0 + w - 1))
        if not _fits(blocked, reserve):
            continue
        _mark(blocked, reserve)
        return {"r0": r0, "c0": c0, "h": h, "w": w, "color": color}
    return None


def _render_input(
    h_grid: int,
    w_grid: int,
    bg: int,
    marker_color: int,
    marked: list[dict],
    distractors: list[dict],
) -> np.ndarray:
    grid = np.full((h_grid, w_grid), bg, dtype=int)
    for obj in distractors:
        r0, c0 = int(obj["r0"]), int(obj["c0"])
        h, w = int(obj["h"]), int(obj["w"])
        grid[r0 : r0 + h, c0 : c0 + w] = int(obj["color"])
    for obj in marked:
        r0, c0 = int(obj["r0"]), int(obj["c0"])
        h, w = int(obj["h"]), int(obj["w"])
        grid[r0 : r0 + h, c0 : c0 + w] = int(obj["color"])
        for mr, mc in obj["markers"]:
            grid[r0 + mr, c0 + mc] = marker_color
    return grid


def _expected_output(
    grid: np.ndarray,
    bg: int,
    marked: list[dict],
) -> np.ndarray:
    out = grid.copy()
    for obj in marked:
        r0, c0 = int(obj["r0"]), int(obj["c0"])
        h, w = int(obj["h"]), int(obj["w"])
        color = int(obj["color"])
        out[r0 : r0 + h, c0 : c0 + w] = color
        bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        _fill_extension(out, bg, color, bbox, str(obj["direction"]), int(obj["count"]))
    return out


def _is_valid_instance(grid: np.ndarray, bg: int, marked_count: int, marker_color: int) -> bool:
    if _mode(grid) != bg:
        return False
    if int(np.sum(grid == marker_color)) < marked_count:
        return False
    comps = _components(grid != bg)
    multi = 0
    for pts in comps:
        colors = {int(grid[r, c]) for r, c in pts}
        if len(colors) > 1:
            multi += 1
    return multi == marked_count


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        h_grid = rng.randint(16, 30)
        w_grid = rng.randint(16, 30)
        marked_count = rng.randint(1, 3)
        distractor_count = rng.randint(0, 2)

        palette = list(range(10))
        rng.shuffle(palette)
        bg = palette.pop()
        marker_color = palette.pop()
        shape_colors = [palette.pop() for _ in range(marked_count + distractor_count)]

        blocked = np.zeros((h_grid, w_grid), dtype=bool)
        marked: list[dict] = []
        distractors: list[dict] = []

        ok = True
        for idx in range(marked_count):
            obj = _place_marked_object(blocked, h_grid, w_grid, shape_colors[idx], marker_color, rng)
            if obj is None:
                ok = False
                break
            marked.append(obj)
        if not ok:
            continue

        for idx in range(distractor_count):
            obj = _place_distractor(blocked, h_grid, w_grid, shape_colors[marked_count + idx], rng)
            if obj is None:
                ok = False
                break
            distractors.append(obj)
        if not ok:
            continue

        grid = _render_input(h_grid, w_grid, bg, marker_color, marked, distractors)
        expected = _expected_output(grid, bg, marked)
        if not _is_valid_instance(grid, bg, marked_count, marker_color):
            continue
        if not np.array_equal(main(grid), expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC instance")
