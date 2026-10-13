import random
from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_RETRIES = 600
PLACEMENT_RETRIES = 250
GRID_SIDE_MIN = 11
GRID_SIDE_MAX = 20
RECT_SIDE_SHORT_MAX = 4
RECT_SIDE_LONG_MAX = 7
INPUT_GAP = 1
MODES = ("plain", "clip_only", "overlap_only", "both")


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[tuple[int, np.ndarray]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, np.ndarray]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))
            components.append((color, np.array(cells, dtype=int)))
    return components


def _extract_rectangles(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int, int, int]]]:
    bg = _most_common_color(grid)
    rects: list[tuple[int, int, int, int, int]] = []

    for color, coords in _find_components(grid, bg):
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert height != width, "rectangles must be non-square"
        assert int(coords.shape[0]) == height * width, "foreground object must be a solid rectangle"
        assert np.all(patch == color), "rectangle bbox must be monochrome"
        rects.append((r0, c0, height, width, color))

    assert rects, "input must contain at least one rectangle"
    return bg, rects


def _swapped_bbox(rect: tuple[int, int, int, int, int], grid_h: int, grid_w: int) -> tuple[int, int, int, int, bool]:
    r0, c0, height, width, _color = rect
    out_h, out_w = width, height
    r1 = min(grid_h, r0 + out_h)
    c1 = min(grid_w, c0 + out_w)
    clipped = (r1 - r0 != out_h) or (c1 - c0 != out_w)
    return r0, c0, r1, c1, clipped


def _render_swapped(
    rects: list[tuple[int, int, int, int, int]],
    grid_h: int,
    grid_w: int,
    bg: int,
) -> np.ndarray:
    out = np.full((grid_h, grid_w), bg, dtype=int)

    # Larger rectangles are painted first so smaller ones remain visible on top.
    ordered = sorted(rects, key=lambda rect: (-(rect[2] * rect[3]), rect[0], rect[1], rect[4]))
    for rect in ordered:
        r0, c0, r1, c1, _clipped = _swapped_bbox(rect, grid_h, grid_w)
        if r0 < r1 and c0 < c1:
            out[r0:r1, c0:c1] = rect[4]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    grid_h, grid_w = g.shape
    bg, rects = _extract_rectangles(g)
    return _render_swapped(rects, grid_h, grid_w, bg)


def _input_box(rect: tuple[int, int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, height, width, _color = rect
    return r0, c0, r0 + height, c0 + width


def _boxes_too_close(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], gap: int) -> bool:
    return not (
        box_a[2] + gap <= box_b[0]
        or box_b[2] + gap <= box_a[0]
        or box_a[3] + gap <= box_b[1]
        or box_b[3] + gap <= box_a[1]
    )


def _can_place_input(
    rect: tuple[int, int, int, int, int],
    rects: list[tuple[int, int, int, int, int]],
    grid_h: int,
    grid_w: int,
) -> bool:
    r0, c0, height, width, _color = rect
    if r0 < 0 or c0 < 0 or r0 + height > grid_h or c0 + width > grid_w:
        return False
    candidate_box = _input_box(rect)
    for other in rects:
        if _boxes_too_close(candidate_box, _input_box(other), INPUT_GAP):
            return False
    return True


def _scene_properties(
    rects: list[tuple[int, int, int, int, int]],
    grid_h: int,
    grid_w: int,
) -> dict[str, int | bool]:
    cover = np.zeros((grid_h, grid_w), dtype=int)
    has_clip = False
    for rect in rects:
        r0, c0, r1, c1, clipped = _swapped_bbox(rect, grid_h, grid_w)
        has_clip = has_clip or clipped
        if r0 < r1 and c0 < c1:
            cover[r0:r1, c0:c1] += 1
    max_stack = int(cover.max()) if cover.size else 0
    overlap_cells = int(np.count_nonzero(cover > 1))
    return {
        "has_clip": bool(has_clip),
        "has_overlap": bool(overlap_cells > 0),
        "max_stack": max_stack,
        "overlap_cells": overlap_cells,
    }


def _matches_mode(props: dict[str, int | bool], mode: str) -> bool:
    has_clip = bool(props["has_clip"])
    has_overlap = bool(props["has_overlap"])
    if mode == "plain":
        return (not has_clip) and (not has_overlap)
    if mode == "clip_only":
        return has_clip and (not has_overlap)
    if mode == "overlap_only":
        return (not has_clip) and has_overlap
    if mode == "both":
        return has_clip and has_overlap
    raise ValueError(f"unknown mode: {mode}")


def _sample_oriented_dims(grid_h: int, grid_w: int, orientation: str) -> tuple[int, int] | None:
    short_limit = min(RECT_SIDE_SHORT_MAX, grid_h, grid_w)
    if short_limit < 2:
        return None

    for _ in range(50):
        short = random.randint(2, short_limit)
        if orientation == "wide":
            long_limit = min(RECT_SIDE_LONG_MAX, grid_w)
            if short + 1 > long_limit or short > grid_h:
                continue
            return short, random.randint(short + 1, long_limit)
        if orientation == "tall":
            long_limit = min(RECT_SIDE_LONG_MAX, grid_h)
            if short + 1 > long_limit or short > grid_w:
                continue
            return random.randint(short + 1, long_limit), short
        raise ValueError(f"unknown orientation: {orientation}")
    return None


def _sample_random_rect(
    grid_h: int,
    grid_w: int,
    color: int,
    rects: list[tuple[int, int, int, int, int]],
    used_areas: set[int],
) -> tuple[int, int, int, int, int] | None:
    for _ in range(PLACEMENT_RETRIES):
        orientation = random.choice(("wide", "tall"))
        dims = _sample_oriented_dims(grid_h, grid_w, orientation)
        if dims is None:
            continue
        height, width = dims
        area = height * width
        if area in used_areas:
            continue
        r0 = random.randint(0, grid_h - height)
        c0 = random.randint(0, grid_w - width)
        rect = (r0, c0, height, width, color)
        if _can_place_input(rect, rects, grid_h, grid_w):
            return rect
    return None


def _sample_overlap_pair(
    grid_h: int,
    grid_w: int,
    color_a: int,
    color_b: int,
    rects: list[tuple[int, int, int, int, int]],
    used_areas: set[int],
) -> list[tuple[int, int, int, int, int]] | None:
    for _ in range(PLACEMENT_RETRIES):
        dims_a = _sample_oriented_dims(grid_h, grid_w, "wide")
        dims_b = _sample_oriented_dims(grid_h, grid_w, "tall")
        if dims_a is None or dims_b is None:
            continue
        h1, w1 = dims_a
        h2, w2 = dims_b
        if h1 * w1 in used_areas or h2 * w2 in used_areas or h1 * w1 == h2 * w2:
            continue
        if w1 <= h1 + INPUT_GAP:
            continue
        if h2 > grid_w:
            continue

        r1_max = min(grid_h - w1, grid_h - h1 - INPUT_GAP - h2)
        if r1_max < 0:
            continue
        r1 = random.randint(0, r1_max)
        c1_max = grid_w - w1
        if c1_max < 0:
            continue
        c1 = random.randint(0, c1_max)

        r2_min = r1 + h1 + INPUT_GAP
        r2_max = min(grid_h - h2, r1 + w1 - 1)
        if r2_min > r2_max:
            continue
        r2 = random.randint(r2_min, r2_max)

        c2_min = c1
        c2_max = min(grid_w - h2, c1 + h1 - 1)
        if c2_min > c2_max:
            continue
        c2 = random.randint(c2_min, c2_max)

        rect_a = (r1, c1, h1, w1, color_a)
        rect_b = (r2, c2, h2, w2, color_b)
        if not _can_place_input(rect_a, rects, grid_h, grid_w):
            continue
        if not _can_place_input(rect_b, rects + [rect_a], grid_h, grid_w):
            continue
        props = _scene_properties(rects + [rect_a, rect_b], grid_h, grid_w)
        if bool(props["has_overlap"]) and not bool(props["has_clip"]) and int(props["max_stack"]) <= 2:
            return [rect_a, rect_b]
    return None


def _sample_clipping_rect(
    grid_h: int,
    grid_w: int,
    color: int,
    rects: list[tuple[int, int, int, int, int]],
    used_areas: set[int],
) -> tuple[int, int, int, int, int] | None:
    for _ in range(PLACEMENT_RETRIES):
        orientation = random.choice(("wide", "tall"))
        dims = _sample_oriented_dims(grid_h, grid_w, orientation)
        if dims is None:
            continue
        height, width = dims
        area = height * width
        if area in used_areas:
            continue

        if orientation == "tall":
            c_min = max(0, grid_w - height + 1)
            c_max = grid_w - width
            r_min = 0
            r_max = grid_h - height
        else:
            r_min = max(0, grid_h - width + 1)
            r_max = grid_h - height
            c_min = 0
            c_max = grid_w - width

        if r_min > r_max or c_min > c_max:
            continue

        r0 = random.randint(r_min, r_max)
        c0 = random.randint(c_min, c_max)
        rect = (r0, c0, height, width, color)
        if not _can_place_input(rect, rects, grid_h, grid_w):
            continue
        props = _scene_properties(rects + [rect], grid_h, grid_w)
        if bool(props["has_clip"]) and int(props["max_stack"]) <= 2:
            return rect
    return None


def _render_input(
    rects: list[tuple[int, int, int, int, int]],
    grid_h: int,
    grid_w: int,
    bg: int,
) -> np.ndarray:
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    for r0, c0, height, width, color in rects:
        grid[r0 : r0 + height, c0 : c0 + width] = color
    return grid


def _final_scene_ok(
    rects: list[tuple[int, int, int, int, int]],
    grid_h: int,
    grid_w: int,
    mode: str,
) -> bool:
    props = _scene_properties(rects, grid_h, grid_w)
    if not _matches_mode(props, mode):
        return False
    if int(props["max_stack"]) > 2:
        return False
    if bool(props["has_overlap"]) and int(props["overlap_cells"]) > 12:
        return False

    areas = [height * width for _r0, _c0, height, width, _color in rects]
    if len(set(areas)) != len(areas):
        return False
    if not any(height > width for _r0, _c0, height, width, _color in rects):
        return False
    if not any(height < width for _r0, _c0, height, width, _color in rects):
        return False

    bg_cells = grid_h * grid_w - sum(areas)
    if bg_cells <= max(areas):
        return False
    return True


def _build_scene(mode: str) -> np.ndarray | None:
    grid_h = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    grid_w = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    bg = random.choice((0, 1, 2))
    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)

    if mode == "both":
        target_count = random.randint(3, 5)
    else:
        target_count = random.randint(2, 5)
    colors = colors[:target_count]

    rects: list[tuple[int, int, int, int, int]] = []
    used_areas: set[int] = set()
    color_index = 0

    if mode in ("overlap_only", "both"):
        pair = _sample_overlap_pair(
            grid_h,
            grid_w,
            colors[color_index],
            colors[color_index + 1],
            rects,
            used_areas,
        )
        if pair is None:
            return None
        rects.extend(pair)
        used_areas.update(rect[2] * rect[3] for rect in pair)
        color_index += 2

    if mode in ("clip_only", "both"):
        clip_rect = _sample_clipping_rect(
            grid_h,
            grid_w,
            colors[color_index],
            rects,
            used_areas,
        )
        if clip_rect is None:
            return None
        rects.append(clip_rect)
        used_areas.add(clip_rect[2] * clip_rect[3])
        color_index += 1

    while len(rects) < target_count:
        color = colors[color_index]
        color_index += 1
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            rect = _sample_random_rect(grid_h, grid_w, color, rects, used_areas)
            if rect is None:
                continue
            trial = rects + [rect]
            props = _scene_properties(trial, grid_h, grid_w)
            if mode in ("plain", "overlap_only") and bool(props["has_clip"]):
                continue
            if mode in ("plain", "clip_only") and bool(props["has_overlap"]):
                continue
            if int(props["max_stack"]) > 2:
                continue
            rects = trial
            used_areas.add(rect[2] * rect[3])
            placed = True
            break
        if not placed:
            return None

    if not _final_scene_ok(rects, grid_h, grid_w, mode):
        return None

    grid = _render_input(rects, grid_h, grid_w, bg)
    if np.array_equal(grid, _render_swapped(rects, grid_h, grid_w, bg)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        mode = random.choice(MODES)
        grid = _build_scene(mode)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
