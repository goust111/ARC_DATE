import random
from collections import deque

import numpy as np


BACKGROUND = 0
OUTLINE = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ("top", "bottom", "left", "right")

SHAPE_COUNT_RANGE = (2, 4)
INNER_H_RANGE = (2, 5)
INNER_W_RANGE = (2, 6)
LEGEND_HEIGHT_CHOICES = (2, 3)
LEGEND_GAP = 1
GRID_H_MAX = 22
GRID_W_LEFT_MAX = 22
GRID_W_RIGHT_RANGE = (6, 8)
PLACEMENT_RETRIES = 200
GEN_ATTEMPTS = 400


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_divider(grid: np.ndarray, bg: int) -> tuple[int, int]:
    center = (grid.shape[1] - 1) / 2.0
    candidates: list[tuple[float, int, int]] = []
    for c in range(1, grid.shape[1] - 1):
        column = grid[:, c]
        if int(column[0]) == bg:
            continue
        if np.all(column == column[0]):
            candidates.append((abs(c - center), c, int(column[0])))
    assert candidates, "expected a full-height monochrome divider column"
    _, divider_col, divider_color = min(candidates)
    return divider_col, divider_color


def _legend_mapping(right: np.ndarray, bg: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for comp in _components(right != bg):
        rows = comp[:, 0]
        cols = comp[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        block = right[r0 : r1 + 1, c0 : c1 + 1]
        if block.shape[1] != 2 or block.shape[0] < 2:
            continue
        if not np.all(block != bg):
            continue
        if not np.all(block[:, 0] == block[0, 0]):
            continue
        if not np.all(block[:, 1] == block[0, 1]):
            continue
        p = int(block[0, 0])
        q = int(block[0, 1])
        if p == q:
            continue
        assert p not in mapping, "legend keys must be unique"
        mapping[p] = q
    assert mapping, "expected at least one legend block on the right side"
    return mapping


def _frame_bboxes(left: np.ndarray, outline: int, bg: int) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(left == outline):
        rows = comp[:, 0]
        cols = comp[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 4 or width < 4:
            continue

        patch = left[r0 : r1 + 1, c0 : c1 + 1]
        perimeter = np.zeros_like(patch, dtype=bool)
        perimeter[0, :] = True
        perimeter[-1, :] = True
        perimeter[:, 0] = True
        perimeter[:, -1] = True

        if not np.all(patch[perimeter] == outline):
            continue
        if np.any(patch[~perimeter] != bg):
            continue
        if int(np.count_nonzero(patch == outline)) != 2 * (height + width) - 4:
            continue
        frames.append((r0, c0, r1, c1))

    assert frames, "expected one or more outlined rectangles on the left side"
    frames.sort()
    return frames


def _marker_for_frame(
    left: np.ndarray,
    bbox: tuple[int, int, int, int],
    outline: int,
    bg: int,
    mapping: dict[int, int],
) -> tuple[str, int]:
    r0, c0, r1, c1 = bbox
    candidates: list[tuple[str, int]] = []

    if r0 > 0:
        strip = left[r0 - 1, c0 : c1 + 1]
        color = int(strip[0])
        if color not in {bg, outline} and color in mapping and np.all(strip == color):
            candidates.append(("top", color))
    if r1 + 1 < left.shape[0]:
        strip = left[r1 + 1, c0 : c1 + 1]
        color = int(strip[0])
        if color not in {bg, outline} and color in mapping and np.all(strip == color):
            candidates.append(("bottom", color))
    if c0 > 0:
        strip = left[r0 : r1 + 1, c0 - 1]
        color = int(strip[0])
        if color not in {bg, outline} and color in mapping and np.all(strip == color):
            candidates.append(("left", color))
    if c1 + 1 < left.shape[1]:
        strip = left[r0 : r1 + 1, c1 + 1]
        color = int(strip[0])
        if color not in {bg, outline} and color in mapping and np.all(strip == color):
            candidates.append(("right", color))

    assert len(candidates) == 1, "each frame must have exactly one marker strip"
    return candidates[0]


def _fill_interior(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    side: str,
    p: int,
    q: int,
) -> None:
    r0, c0, r1, c1 = bbox
    if side == "top":
        for idx, r in enumerate(range(r0 + 1, r1)):
            out[r, c0 + 1 : c1] = p if idx % 2 == 0 else q
    elif side == "bottom":
        for idx, r in enumerate(range(r1 - 1, r0, -1)):
            out[r, c0 + 1 : c1] = p if idx % 2 == 0 else q
    elif side == "left":
        for idx, c in enumerate(range(c0 + 1, c1)):
            out[r0 + 1 : r1, c] = p if idx % 2 == 0 else q
    else:
        for idx, c in enumerate(range(c1 - 1, c0, -1)):
            out[r0 + 1 : r1, c] = p if idx % 2 == 0 else q


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each outlined rectangle using a direction marker and a right-side color legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _mode_color(g)
    divider_col, outline = _find_divider(g, bg)
    left = g[:, :divider_col]
    right = g[:, divider_col + 1 :]

    mapping = _legend_mapping(right, bg)
    frames = _frame_bboxes(left, outline, bg)

    out = g.copy()
    for bbox in frames:
        side, p = _marker_for_frame(left, bbox, outline, bg, mapping)
        q = mapping[p]
        _fill_interior(out, bbox, side, p, q)
    return out


def _draw_legend(
    grid: np.ndarray,
    divider_col: int,
    entries: list[tuple[int, int]],
    heights: list[int],
) -> None:
    total_height = sum(heights) + LEGEND_GAP * (len(heights) - 1)
    start_row = random.randint(1, grid.shape[0] - total_height - 1)
    start_col = random.randint(divider_col + 2, grid.shape[1] - 3)
    row = start_row
    shuffled = entries[:]
    random.shuffle(shuffled)
    for (p, q), height in zip(shuffled, heights):
        grid[row : row + height, start_col] = p
        grid[row : row + height, start_col + 1] = q
        row += height + LEGEND_GAP


def _shape_used_bbox(
    r0: int,
    c0: int,
    outer_h: int,
    outer_w: int,
    side: str,
) -> tuple[int, int, int, int]:
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    if side == "top":
        r0 -= 1
    elif side == "bottom":
        r1 += 1
    elif side == "left":
        c0 -= 1
    else:
        c1 += 1
    return r0, c0, r1, c1


def _place_shape(
    grid: np.ndarray,
    occupancy: np.ndarray,
    left_width: int,
    side: str,
    marker_color: int,
    inner_h: int,
    inner_w: int,
) -> bool:
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    row_min = 1 if side == "top" else 0
    row_max = grid.shape[0] - outer_h - (1 if side == "bottom" else 0)
    col_min = 1 if side == "left" else 0
    col_max = left_width - outer_w - (1 if side == "right" else 0)
    if row_max < row_min or col_max < col_min:
        return False

    for _ in range(PLACEMENT_RETRIES):
        r0 = random.randint(row_min, row_max)
        c0 = random.randint(col_min, col_max)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1

        ur0, uc0, ur1, uc1 = _shape_used_bbox(r0, c0, outer_h, outer_w, side)
        pr0 = max(0, ur0 - 1)
        pc0 = max(0, uc0 - 1)
        pr1 = min(grid.shape[0] - 1, ur1 + 1)
        pc1 = min(left_width - 1, uc1 + 1)
        if np.any(occupancy[pr0 : pr1 + 1, pc0 : pc1 + 1]):
            continue

        grid[r0, c0 : c1 + 1] = OUTLINE
        grid[r1, c0 : c1 + 1] = OUTLINE
        grid[r0 : r1 + 1, c0] = OUTLINE
        grid[r0 : r1 + 1, c1] = OUTLINE

        if side == "top":
            grid[r0 - 1, c0 : c1 + 1] = marker_color
        elif side == "bottom":
            grid[r1 + 1, c0 : c1 + 1] = marker_color
        elif side == "left":
            grid[r0 : r1 + 1, c0 - 1] = marker_color
        else:
            grid[r0 : r1 + 1, c1 + 1] = marker_color

        occupancy[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    shape_count = random.randint(*SHAPE_COUNT_RANGE)
    legend_heights = [random.choice(LEGEND_HEIGHT_CHOICES) for _ in range(shape_count)]
    legend_total_height = sum(legend_heights) + LEGEND_GAP * (shape_count - 1)

    height_min = max(14, legend_total_height + 2)
    if height_min > GRID_H_MAX:
        return None
    height = random.randint(height_min, GRID_H_MAX)

    left_width_min = max(14, 4 * shape_count + 4)
    left_width_max = max(left_width_min, GRID_W_LEFT_MAX)
    if left_width_min > left_width_max:
        return None
    left_width = random.randint(left_width_min, left_width_max)
    right_width = random.randint(*GRID_W_RIGHT_RANGE)

    grid = np.full((height, left_width + 1 + right_width), BACKGROUND, dtype=int)
    divider_col = left_width
    grid[:, divider_col] = OUTLINE

    marker_colors = random.sample(MARKER_COLORS, shape_count)
    fill_candidates = [color for color in MARKER_COLORS if color not in marker_colors]
    fill_colors = random.sample(fill_candidates, shape_count)
    legend_entries = list(zip(marker_colors, fill_colors))
    _draw_legend(grid, divider_col, legend_entries, legend_heights)

    directions = random.sample(DIRECTIONS, shape_count)
    specs = []
    for marker_color, side in zip(marker_colors, directions):
        inner_h = random.randint(*INNER_H_RANGE)
        inner_w = random.randint(*INNER_W_RANGE)
        specs.append(
            {
                "marker_color": marker_color,
                "side": side,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "priority": (inner_h + 2) * (inner_w + 2),
            }
        )
    specs.sort(key=lambda spec: int(spec["priority"]), reverse=True)

    occupancy = np.zeros((height, left_width), dtype=bool)
    for spec in specs:
        ok = _place_shape(
            grid=grid,
            occupancy=occupancy,
            left_width=left_width,
            side=str(spec["side"]),
            marker_color=int(spec["marker_color"]),
            inner_h=int(spec["inner_h"]),
            inner_w=int(spec["inner_w"]),
        )
        if not ok:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
