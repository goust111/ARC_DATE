from __future__ import annotations

import random

import numpy as np


BLACK = 0
FRAME_COLOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 500

BBox = tuple[int, int, int, int]


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _border_points(box: BBox) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _paint_border(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _components_by_color(grid: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append((color, coords))
    return comps


def _frame_bbox(coords: list[tuple[int, int]]) -> BBox | None:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None
    border = set(_border_points((r0, c0, r1, c1)))
    if len(border) != len(coords):
        return None
    if set(coords) != border:
        return None
    return (r0, c0, r1, c1)


def _contains_with_gap(outer: BBox, inner: BBox) -> bool:
    return (
        outer[0] + 2 <= inner[0]
        and outer[1] + 2 <= inner[1]
        and inner[2] <= outer[2] - 2
        and inner[3] <= outer[3] - 2
    )


def _separated_with_gap(a: BBox, b: BBox) -> bool:
    return (
        a[2] + 1 < b[0]
        or b[2] + 1 < a[0]
        or a[3] + 1 < b[1]
        or b[3] + 1 < a[1]
    )


def _compatible(candidate: BBox, boxes: list[BBox]) -> bool:
    for other in boxes:
        if (
            _contains_with_gap(candidate, other)
            or _contains_with_gap(other, candidate)
            or _separated_with_gap(candidate, other)
        ):
            continue
        return False
    return True


def _depths(boxes: list[BBox]) -> list[int]:
    return [1 + sum(_contains_with_gap(other, box) for other in boxes if other != box) for box in boxes]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each rectangular frame according to the depth-indexed legend in the left column.

    Input structure:
    - Black background.
    - Some non-black connected components are 1-pixel-thick hollow rectangular
      frames; nested containment defines their depths.
    - The leftmost column contains isolated colored legend pixels ordered from
      top to bottom; they map depths 1, 2, 3, ... to output colors.
    - Any remaining non-black pixels are distractors and are not part of frames.

    Rule:
    - Read the non-black pixels in column 0 from top to bottom to obtain the
      depth-to-color legend.
    - Detect every connected component whose cells exactly match the border of
      an axis-aligned hollow rectangle.
    - Compute each frame's depth from strict containment: outermost frames have
      depth 1, frames inside one frame have depth 2, and so on.
    - Output only the frames, recolored by their depth legend entry; clear the
      left-column legend and all distractor pixels to black.

    Key insight:
    The left column is a depth legend, but using it correctly first requires
    inferring the nesting tree of the rectangular frame components.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = BLACK
    h, _w = g.shape
    legend = [int(g[r, 0]) for r in range(h) if int(g[r, 0]) != bg]
    assert legend, "expected at least one legend color in the leftmost column"

    frames: list[BBox] = []
    for _color, coords in _components_by_color(g, bg):
        box = _frame_bbox(coords)
        if box is not None:
            frames.append(box)
    assert frames, "expected at least one rectangular frame"

    depths = _depths(frames)
    max_depth = max(depths)
    assert len(legend) >= max_depth, "legend must cover every frame depth"

    out = np.full_like(g, bg)
    for box, depth in zip(frames, depths):
        _paint_border(out, box, legend[depth - 1])
    return out


def _sample_chain() -> tuple[int, int, list[BBox]] | None:
    depth = random.randint(1, 4)
    inner_h = random.randint(3, 7)
    inner_w = random.randint(3, 8)
    inward_gaps = [
        (
            random.randint(2, 3),
            random.randint(2, 3),
            random.randint(2, 3),
            random.randint(2, 3),
        )
        for _ in range(depth - 1)
    ]
    outer_h = inner_h + sum(top + bottom for top, bottom, _left, _right in inward_gaps)
    outer_w = inner_w + sum(left + right for _top, _bottom, left, right in inward_gaps)

    min_h = max(10, outer_h + 2)
    min_w = max(10, outer_w + 3)
    if min_h > 30 or min_w > 30:
        return None

    h = random.randint(min_h, min(30, outer_h + 10))
    w = random.randint(min_w, min(30, outer_w + 12))
    r0 = random.randint(1, h - outer_h - 1)
    c0 = random.randint(2, w - outer_w - 1)
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1

    boxes = [(r0, c0, r1, c1)]
    cr0, cc0, cr1, cc1 = r0, c0, r1, c1
    for top, bottom, left, right in inward_gaps:
        cr0 += top
        cc0 += left
        cr1 -= bottom
        cc1 -= right
        boxes.append((cr0, cc0, cr1, cc1))
    return h, w, boxes


def _candidate_box(h: int, w: int, boxes: list[BBox]) -> BBox | None:
    depths = _depths(boxes)
    eligible_parents = [
        box
        for box, depth in zip(boxes, depths)
        if depth < 4 and (box[2] - box[0] + 1) >= 7 and (box[3] - box[1] + 1) >= 7
    ]
    modes = ["root"]
    if eligible_parents:
        modes.extend(["inside", "inside"])

    for _ in range(80):
        mode = random.choice(modes)
        if mode == "inside":
            parent = random.choice(eligible_parents)
            max_h = parent[2] - parent[0] - 3
            max_w = parent[3] - parent[1] - 3
            if max_h < 3 or max_w < 3:
                continue
            box_h = random.randint(3, min(10, max_h))
            box_w = random.randint(3, min(10, max_w))
            r0 = random.randint(parent[0] + 2, parent[2] - box_h - 1)
            c0 = random.randint(parent[1] + 2, parent[3] - box_w - 1)
        else:
            box_h = random.randint(3, min(12, h - 2))
            box_w = random.randint(3, min(12, w - 3))
            r0 = random.randint(1, h - box_h - 1)
            c0 = random.randint(2, w - box_w - 1)
        candidate = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        if not _compatible(candidate, boxes):
            continue
        new_boxes = boxes + [candidate]
        if max(_depths(new_boxes)) > 4:
            continue
        return candidate
    return None


def _spaced_rows(h: int, count: int) -> list[int] | None:
    for _ in range(100):
        rows = sorted(random.sample(range(h), count))
        if all(b - a >= 2 for a, b in zip(rows, rows[1:])):
            return rows
    return None


def _noise_positions(grid: np.ndarray, boxes: list[BBox]) -> list[tuple[int, int]]:
    h, w = grid.shape
    cells: list[tuple[int, int]] = []
    for r0, c0, r1, c1 in boxes:
        for r in range(r0 + 2, r1 - 1):
            for c in range(c0 + 2, c1 - 1):
                if int(grid[r, c]) != BLACK:
                    continue
                isolated = True
                for rr in range(max(0, r - 1), min(h, r + 2)):
                    for cc in range(max(0, c - 1), min(w, c + 2)):
                        if (rr, cc) == (r, c):
                            continue
                        if int(grid[rr, cc]) != BLACK:
                            isolated = False
                            break
                    if not isolated:
                        break
                if isolated:
                    cells.append((r, c))
    random.shuffle(cells)
    return cells


def _expected_output(shape: tuple[int, int], boxes: list[BBox], legend_colors: list[int]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for box, depth in zip(boxes, _depths(boxes)):
        _paint_border(out, box, legend_colors[depth - 1])
    return out


def _gen_attempt() -> np.ndarray | None:
    base = _sample_chain()
    if base is None:
        return None
    h, w, boxes = base

    target_total = random.randint(len(boxes), min(8, len(boxes) + 4))
    while len(boxes) < target_total:
        candidate = _candidate_box(h, w, boxes)
        if candidate is None:
            break
        boxes.append(candidate)

    max_depth = max(_depths(boxes))
    legend_rows = _spaced_rows(h, max_depth)
    if legend_rows is None:
        return None
    legend_colors = random.sample(list(COLOR_POOL), max_depth)

    grid = np.full((h, w), BLACK, dtype=int)
    for box in boxes:
        _paint_border(grid, box, FRAME_COLOR)
    for row, color in zip(legend_rows, legend_colors):
        grid[row, 0] = color

    noise_palette = [c for c in COLOR_POOL if c not in legend_colors]
    if not noise_palette:
        noise_palette = list(COLOR_POOL)
    noise_slots = _noise_positions(grid, boxes)
    noise_count = random.randint(0, min(6, len(noise_slots)))
    for r, c in noise_slots[:noise_count]:
        grid[r, c] = random.choice(noise_palette)

    expected = _expected_output((h, w), boxes, legend_colors)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame ARC input")
