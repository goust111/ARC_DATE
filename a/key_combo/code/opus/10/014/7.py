from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GREEN = 3
YELLOW = 4
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [pt[0] for pt in coords]
            cols = [pt[1] for pt in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _marker_positions(bbox: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = bbox
    return ((r0 - 1, c0 - 1), (r0 - 1, c1 + 1), (r1 + 1, c0 - 1), (r1 + 1, c1 + 1))


def _find_border_segment(grid: np.ndarray) -> dict:
    h, w = grid.shape
    border_pts = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if int(grid[r, c]) == YELLOW and (r == 0 or r == h - 1 or c == 0 or c == w - 1)
    ]
    assert border_pts
    rows = {r for r, _ in border_pts}
    cols = {c for _, c in border_pts}
    if len(rows) == 1:
        row = next(iter(rows))
        span0, span1 = min(cols), max(cols)
        return {
            "orient": "horizontal",
            "start": row,
            "dir": 1 if row == 0 else -1,
            "span0": span0,
            "span1": span1,
            "limit": h,
        }
    assert len(cols) == 1
    col = next(iter(cols))
    span0, span1 = min(rows), max(rows)
    return {
        "orient": "vertical",
        "start": col,
        "dir": 1 if col == 0 else -1,
        "span0": span0,
        "span1": span1,
        "limit": w,
    }


def _apply_stripes(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    bg = _get_bg(out)
    seg = _find_border_segment(out)
    cycle = (YELLOW, YELLOW, GREEN)
    step = 0
    while True:
        pos = int(seg["start"]) + int(seg["dir"]) * (2 * step)
        if not (0 <= pos < int(seg["limit"])):
            break
        color = cycle[step % 3]
        if seg["orient"] == "horizontal":
            if color == GREEN:
                row_mask = out[pos, :] != bg
                out[pos, row_mask] = GREEN
            out[pos, seg["span0"] : seg["span1"] + 1] = color
            gap = int(seg["start"]) + int(seg["dir"]) * (2 * step + 1)
            if 0 <= gap < out.shape[0]:
                out[gap, seg["span0"] : seg["span1"] + 1] = bg
        else:
            if color == GREEN:
                col_mask = out[:, pos] != bg
                out[col_mask, pos] = GREEN
            out[seg["span0"] : seg["span1"] + 1, pos] = color
            gap = int(seg["start"]) + int(seg["dir"]) * (2 * step + 1)
            if 0 <= gap < out.shape[1]:
                out[seg["span0"] : seg["span1"] + 1, gap] = bg
        step += 1
    return out


def _center_statuses(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    h, w = g.shape
    statuses: list[dict] = []
    for comp in _components(g, bg):
        r0, c0, r1, c1 = comp["bbox"]
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        if comp["area"] < 4:
            continue
        if not (0 < r0 and 0 < c0 and r1 < h - 1 and c1 < w - 1):
            continue
        if not (3 <= box_h <= 5 and 3 <= box_w <= 5):
            continue
        marker_pts = _marker_positions(comp["bbox"])
        marker_colors = [int(g[r, c]) for r, c in marker_pts]
        if any(color == bg for color in marker_colors):
            continue
        statuses.append(
            {
                "bbox": comp["bbox"],
                "center_color": int(comp["color"]),
                "marker_colors": marker_colors,
                "uniform": len(set(marker_colors)) == 1,
                "marker_color": marker_colors[0],
            }
        )
    return statuses


def _matching_centers(grid: np.ndarray) -> list[dict]:
    return [
        status
        for status in _center_statuses(grid)
        if status["uniform"] and status["marker_color"] == status["center_color"]
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the unique center whose corner markers only become color-matched after stripe propagation.

    Input structure:
    - The grid has a black background, one yellow border segment, and 3-4 marker clusters.
    - Each cluster consists of one monochromatic center object and four single-pixel corner markers one cell diagonally outside its bbox corners.
    - Initially, no cluster has marker color equal to center color.

    Rule:
    - Propagate stripes from the yellow border segment toward the opposite edge in a Y, Y, G cycle with a one-cell gap between stripes.
    - Yellow stripes paint only inside the segment span, while each green stripe recolors every non-background cell on its full row or column to green.
    - After propagation, find the unique cluster whose four markers are still uniform and now match the center color, then crop that center bbox.

    Key insight:
    - The border pattern is not decorative: the green stripes rewrite marker colors first, and only then can the correct cluster be identified.
    """
    transformed = _apply_stripes(np.asarray(grid, dtype=int))
    matches = _matching_centers(transformed)
    assert len(matches) == 1
    r0, c0, r1, c1 = matches[0]["bbox"]
    return transformed[r0 : r1 + 1, c0 : c1 + 1].copy()


def _outer_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 - 1, c0 - 1, r1 + 1, c1 + 1)


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _draw_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    tr, tc = end
    mask[r, c] = True
    while (r, c) != (tr, tc):
        vertical = tr != r
        horizontal = tc != c
        if vertical and horizontal:
            if random.random() < 0.5:
                r += 1 if tr > r else -1
            else:
                c += 1 if tc > c else -1
        elif vertical:
            r += 1 if tr > r else -1
        else:
            c += 1 if tc > c else -1
        mask[r, c] = True


def _random_connected_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    anchors = [
        (0, random.randrange(width)),
        (height - 1, random.randrange(width)),
        (random.randrange(height), 0),
        (random.randrange(height), width - 1),
    ]
    unique_anchors: list[tuple[int, int]] = []
    for anchor in anchors:
        if anchor not in unique_anchors:
            unique_anchors.append(anchor)
    mask[unique_anchors[0]] = True
    current = unique_anchors[0]
    for anchor in unique_anchors[1:]:
        _draw_path(mask, current, anchor)
        current = anchor
    min_area = max(height + width - 2, (height * width + 1) // 2)
    target_area = random.randint(min_area, height * width)
    while int(mask.sum()) < target_area:
        frontier = set()
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True
    return mask


def _green_positions(length: int, edge: str) -> list[int]:
    start = 0 if edge in ("top", "left") else length - 1
    direction = 1 if start == 0 else -1
    greens: list[int] = []
    step = 0
    while True:
        pos = start + direction * (2 * step)
        if not (0 <= pos < length):
            break
        if step % 3 == 2:
            greens.append(pos)
        step += 1
    greens.sort()
    return greens


def _line_hit_count(start: int, size: int, green_set: set[int]) -> int:
    return int(start - 1 in green_set) + int(start + size in green_set)


def _sample_bbox(
    rows: int,
    cols: int,
    axis: str,
    green_set: set[int],
    prop_sizes: tuple[int, ...],
    cross_sizes: tuple[int, ...],
    desired_hits: int,
    used_outer: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    prop_choices = list(prop_sizes)
    cross_choices = list(cross_sizes)
    random.shuffle(prop_choices)
    random.shuffle(cross_choices)
    for prop_size in prop_choices:
        axis_len = rows if axis == "horizontal" else cols
        prop_starts = [
            start
            for start in range(2, axis_len - prop_size - 1)
            if _line_hit_count(start, prop_size, green_set) == desired_hits
        ]
        random.shuffle(prop_starts)
        if not prop_starts:
            continue
        for cross_size in cross_choices:
            cross_len = cols if axis == "horizontal" else rows
            cross_starts = list(range(2, cross_len - cross_size - 1))
            random.shuffle(cross_starts)
            for prop_start in prop_starts:
                for cross_start in cross_starts:
                    if axis == "horizontal":
                        bbox = (
                            prop_start,
                            cross_start,
                            prop_start + prop_size - 1,
                            cross_start + cross_size - 1,
                        )
                    else:
                        bbox = (
                            cross_start,
                            prop_start,
                            cross_start + cross_size - 1,
                            prop_start + prop_size - 1,
                        )
                    outer = _outer_bbox(bbox)
                    if all(_boxes_separated(outer, prev) for prev in used_outer):
                        return bbox
    return None


def _render_cluster(grid: np.ndarray, bbox: tuple[int, int, int, int], center_color: int, marker_color: int) -> None:
    r0, c0, r1, c1 = bbox
    mask = _random_connected_mask(r1 - r0 + 1, c1 - c0 + 1)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    patch[mask] = center_color
    for mr, mc in _marker_positions(bbox):
        grid[mr, mc] = marker_color


def _choose_segment_start(
    limit: int,
    seg_len: int,
    must_cover: set[int],
    blocked: set[int],
) -> int | None:
    starts = []
    for start in range(1, limit - seg_len):
        span = set(range(start, start + seg_len))
        if not (span & must_cover):
            continue
        if span & blocked:
            continue
        starts.append(start)
    return random.choice(starts) if starts else None


def _bbox_columns(bbox: tuple[int, int, int, int]) -> set[int]:
    return set(range(bbox[1], bbox[3] + 1))


def _bbox_rows(bbox: tuple[int, int, int, int]) -> set[int]:
    return set(range(bbox[0], bbox[2] + 1))


def _oracle_valid(grid: np.ndarray, target_bbox: tuple[int, int, int, int]) -> bool:
    if _matching_centers(grid):
        return False
    transformed = _apply_stripes(grid)
    matches = _matching_centers(transformed)
    if len(matches) != 1 or matches[0]["bbox"] != target_bbox:
        return False
    statuses = _center_statuses(transformed)
    has_full_false = any(
        status["bbox"] != target_bbox
        and status["uniform"]
        and status["marker_color"] == GREEN
        and status["center_color"] != GREEN
        for status in statuses
    )
    has_partial_green = any(
        status["bbox"] != target_bbox
        and status["center_color"] == GREEN
        and not (status["uniform"] and status["marker_color"] == GREEN)
        for status in statuses
    )
    return has_full_false and has_partial_green


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(19, 24)
    cols = random.randint(19, 24)
    axis = random.choice(("horizontal", "vertical"))
    edge = random.choice(("top", "bottom")) if axis == "horizontal" else random.choice(("left", "right"))
    green_set = set(_green_positions(rows if axis == "horizontal" else cols, edge))
    if len(green_set) < 2:
        return None

    color_pool = [1, 2, 5, 6, 7, 8, 9]
    random.shuffle(color_pool)
    if len(color_pool) < 5:
        return None

    marker_target = color_pool.pop()
    center_false = color_pool.pop()
    marker_false = color_pool.pop()
    marker_partial = color_pool.pop()
    extra_center = color_pool.pop() if color_pool else random.choice([1, 2, 5, 6, 7, 8, 9])
    extra_marker = random.choice([c for c in [1, 2, 5, 6, 7, 8, 9] if c != extra_center])

    used_outer: list[tuple[int, int, int, int]] = []
    specs: list[dict] = []

    target_bbox = _sample_bbox(rows, cols, axis, green_set, (5,), (3, 4), 2, used_outer)
    if target_bbox is None:
        return None
    used_outer.append(_outer_bbox(target_bbox))
    specs.append({"role": "target", "bbox": target_bbox, "center_color": GREEN, "marker_color": marker_target})

    false_bbox = _sample_bbox(rows, cols, axis, green_set, (5,), (3, 4), 2, used_outer)
    if false_bbox is None:
        return None
    used_outer.append(_outer_bbox(false_bbox))
    specs.append({"role": "false_full", "bbox": false_bbox, "center_color": center_false, "marker_color": marker_false})

    partial_bbox = _sample_bbox(rows, cols, axis, green_set, (3, 4), (3, 4), 1, used_outer)
    if partial_bbox is None:
        return None
    used_outer.append(_outer_bbox(partial_bbox))
    specs.append({"role": "partial_green", "bbox": partial_bbox, "center_color": GREEN, "marker_color": marker_partial})

    if random.random() < 0.6:
        extra_bbox = _sample_bbox(rows, cols, axis, green_set, (3, 4, 5), (3, 4), random.choice((0, 1)), used_outer)
        if extra_bbox is None:
            return None
        used_outer.append(_outer_bbox(extra_bbox))
        specs.append({"role": "extra", "bbox": extra_bbox, "center_color": extra_center, "marker_color": extra_marker})

    grid = np.zeros((rows, cols), dtype=int)
    for spec in specs:
        if spec["center_color"] == spec["marker_color"]:
            return None
        _render_cluster(grid, spec["bbox"], spec["center_color"], spec["marker_color"])

    target_markers = _marker_positions(target_bbox)
    if axis == "horizontal":
        target_lines = {pt[1] for pt in target_markers}
        blocked = set()
        for spec in specs:
            blocked |= _bbox_columns(spec["bbox"])
            if spec["role"] != "target":
                blocked |= {pt[1] for pt in _marker_positions(spec["bbox"])}
        seg_len = random.choice((2, 3))
        seg_start = _choose_segment_start(cols, seg_len, target_lines, blocked)
        if seg_start is None:
            return None
        border_row = 0 if edge == "top" else rows - 1
        grid[border_row, seg_start : seg_start + seg_len] = YELLOW
    else:
        target_lines = {pt[0] for pt in target_markers}
        blocked = set()
        for spec in specs:
            blocked |= _bbox_rows(spec["bbox"])
            if spec["role"] != "target":
                blocked |= {pt[0] for pt in _marker_positions(spec["bbox"])}
        seg_len = random.choice((2, 3))
        seg_start = _choose_segment_start(rows, seg_len, target_lines, blocked)
        if seg_start is None:
            return None
        border_col = 0 if edge == "left" else cols - 1
        grid[seg_start : seg_start + seg_len, border_col] = YELLOW

    if not _oracle_valid(grid, target_bbox):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
