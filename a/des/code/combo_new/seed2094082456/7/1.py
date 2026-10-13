from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
DIRS = ("up", "down", "left", "right")

FRAME_COUNT_RANGE = (2, 4)
GRID_SIDE_RANGE = (18, 30)
TOP_RESERVED_RANGE = (3, 5)
LEGEND_GAP_CHOICES = (1, 2)
FRAME_HEIGHT_RANGE = (6, 9)
FRAME_WIDTH_RANGE = (6, 10)
PATTERN_FILL_RANGE = (0.18, 0.45)
PATTERN_MIN_AREA = 3
PATTERN_RETRIES = 60
FRAME_PLACEMENT_RETRIES = 120
GENERATION_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _contiguous_run(bits: np.ndarray) -> tuple[int, int] | None:
    idx = np.flatnonzero(bits)
    if idx.size == 0:
        return None
    if idx.size < 2 or int(idx[-1] - idx[0] + 1) != int(idx.size):
        return (-1, -1)
    return int(idx[0]), int(idx[-1])


def _parse_legend(grid: np.ndarray) -> tuple[dict[int, int], int]:
    nonempty_rows = np.flatnonzero(np.any(grid != BG, axis=1))
    assert nonempty_rows.size > 0, "grid must contain a legend row"
    legend_row = int(nonempty_rows[0])

    cols = np.flatnonzero(grid[legend_row] != BG)
    assert cols.size >= 2 and cols.size % 2 == 0, "legend must contain 1x2 color pairs"

    mapping: dict[int, int] = {}
    for i in range(0, int(cols.size), 2):
        c0 = int(cols[i])
        c1 = int(cols[i + 1])
        assert c1 == c0 + 1, "each legend pair must occupy two adjacent cells"
        src = int(grid[legend_row, c0])
        dst = int(grid[legend_row, c1])
        assert src != BG and dst != BG and src != dst
        assert src not in mapping, "frame colors must map unambiguously"
        mapping[src] = dst
    return mapping, legend_row


def _frame_info(
    grid: np.ndarray,
    frame_component: list[tuple[int, int]],
    frame_color: int,
) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(frame_component)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 4 or width < 4:
        return None

    sub = np.zeros((height, width), dtype=bool)
    for r, c in frame_component:
        sub[r - r0, c - c0] = True

    if np.any(sub[1:-1, 1:-1]):
        return None
    if not (sub[0, 0] and sub[0, -1] and sub[-1, 0] and sub[-1, -1]):
        return None

    missing_by_side = {
        "up": ~sub[0, 1:-1],
        "down": ~sub[-1, 1:-1],
        "left": ~sub[1:-1, 0],
        "right": ~sub[1:-1, -1],
    }
    gap_side = None
    for side, bits in missing_by_side.items():
        run = _contiguous_run(bits)
        if run is None:
            continue
        if run == (-1, -1):
            return None
        if gap_side is not None:
            return None
        gap_side = side
    if gap_side is None:
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or not np.any(interior != BG):
        return None
    if np.any(interior == frame_color):
        return None

    pattern_points = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if int(grid[r, c]) != BG
    ]
    return {
        "color": int(frame_color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap_side": gap_side,
        "frame_points": list(frame_component),
        "pattern_points": pattern_points,
    }


def _project_points(
    points: list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    gap_side: str,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    projected: list[tuple[int, int]] = []
    for r, c in points:
        if gap_side == "up":
            projected.append((r0 - 1, c))
        elif gap_side == "down":
            projected.append((r1 + 1, c))
        elif gap_side == "left":
            projected.append((r, c0 - 1))
        else:
            projected.append((r, c1 + 1))
    return projected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dump each framed interior pattern outside the gap side and recolor it via the top legend.

    Input structure:
    - The background is black.
    - Several one-pixel-thick rectangular frames are scattered in the grid; each frame uses one color and has exactly one contiguous gap on one side.
    - Each frame contains non-black pattern pixels, and the top row contains 1x2 legend pairs whose left cell is a frame color and right cell is that frame's dump color.

    Rule:
    - Read the top legend to map every frame color to its target output color.
    - For each rectangular frame, detect which side contains the gap.
    - Move every interior pattern pixel outward along that side until it lands just beyond the frame boundary, so all dumped pixels align on the exterior row or column beside the gap side.
    - Erase the original legend, frames, and interior patterns, and keep only the dumped pixels recolored with the legend-mapped target color.

    Key insight:
    - Solving requires combining two independent cues: the gap side determines the dump direction, while the separate top legend determines the replacement color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    legend, legend_row = _parse_legend(g)
    work = g.copy()
    work[legend_row, :] = BG

    frames: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(work) if int(v) != BG):
        for comp in _components(work == color):
            info = _frame_info(g, comp, color)
            if info is not None:
                frames.append(info)

    assert frames, "expected at least one valid frame"

    covered = np.zeros_like(g, dtype=bool)
    covered[legend_row, g[legend_row] != BG] = True
    out = np.zeros_like(g)

    for frame in frames:
        frame_color = int(frame["color"])
        assert frame_color in legend, "every frame color must appear in the legend"
        bbox = frame["bbox"]
        gap_side = str(frame["gap_side"])
        pattern_points = list(frame["pattern_points"])
        target = int(legend[frame_color])

        for r, c in frame["frame_points"]:
            covered[r, c] = True
        for r, c in pattern_points:
            covered[r, c] = True

        for rr, cc in _project_points(pattern_points, bbox, gap_side):
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1], "dumped pixels must stay in bounds"
            out[rr, cc] = target

    assert np.array_equal(covered, g != BG), "all non-background pixels must belong to the legend or a frame system"
    return out


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((int(nr), int(nc)))
    return out


def _grow_pattern_mask(height: int, width: int, target_area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    seed = (random.randrange(height), random.randrange(width))
    mask[seed] = True
    cells = {seed}
    while len(cells) < target_area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            frontier.update(_neighbors4(r, c, height, width))
        choices = [cell for cell in frontier if not mask[cell]]
        if not choices:
            break
        cell = random.choice(choices)
        mask[cell] = True
        cells.add(cell)
    return mask


def _random_pattern_mask(height: int, width: int, direction: str) -> np.ndarray | None:
    area = height * width
    lo = max(PATTERN_MIN_AREA, int(math.ceil(area * PATTERN_FILL_RANGE[0])))
    hi = max(lo, int(math.floor(area * PATTERN_FILL_RANGE[1])))
    for _ in range(PATTERN_RETRIES):
        target_area = random.randint(lo, hi)
        mask = _grow_pattern_mask(height, width, target_area)
        if int(mask.sum()) != target_area:
            continue
        rows, cols = np.nonzero(mask)
        if len(set(map(int, rows))) < 2 or len(set(map(int, cols))) < 2:
            continue
        if direction in ("up", "down"):
            if int(np.count_nonzero(np.any(mask, axis=0))) < 2:
                continue
        else:
            if int(np.count_nonzero(np.any(mask, axis=1))) < 2:
                continue
        return mask
    return None


def _sample_gap(span: int) -> tuple[int, int]:
    assert span >= 4
    gap_width = random.randint(2, span - 2)
    gap_start = random.randint(1, span - gap_width - 1)
    return gap_start, gap_width


def _make_frame_patch(
    height: int,
    width: int,
    frame_color: int,
    pattern_color: int,
    direction: str,
    pattern_mask: np.ndarray,
) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color

    if direction in ("up", "down"):
        gap_start, gap_width = _sample_gap(width - 2)
        row = 0 if direction == "up" else height - 1
        patch[row, 1 + gap_start : 1 + gap_start + gap_width] = BG
    else:
        gap_start, gap_width = _sample_gap(height - 2)
        col = 0 if direction == "left" else width - 1
        patch[1 + gap_start : 1 + gap_start + gap_width, col] = BG

    interior = patch[1:-1, 1:-1]
    interior[pattern_mask] = pattern_color
    return patch


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    dst[top : top + h, left : left + w] = patch


def _place_legend(grid: np.ndarray, frame_colors: list[int], target_colors: list[int]) -> None:
    pair_gap = random.choice(LEGEND_GAP_CHOICES)
    total_width = 2 * len(frame_colors) + pair_gap * (len(frame_colors) - 1)
    start_col = random.randint(0, grid.shape[1] - total_width)
    c = start_col
    for frame_color, target_color in zip(frame_colors, target_colors):
        grid[0, c] = frame_color
        grid[0, c + 1] = target_color
        c += 2 + pair_gap


def _attempt_generate() -> np.ndarray | None:
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    frame_colors = random.sample(COLORS, frame_count)
    target_pool = [c for c in COLORS if c not in frame_colors]
    target_colors = random.sample(target_pool, frame_count)

    top_reserved = random.randint(*TOP_RESERVED_RANGE)
    legend_gap = max(LEGEND_GAP_CHOICES)
    legend_width = 2 * frame_count + legend_gap * (frame_count - 1)
    min_h = max(GRID_SIDE_RANGE[0], top_reserved + 7 + frame_count)
    min_w = max(GRID_SIDE_RANGE[0], legend_width + 4 + frame_count)
    if min_h > GRID_SIDE_RANGE[1] or min_w > GRID_SIDE_RANGE[1]:
        return None

    grid_h = random.randint(min_h, GRID_SIDE_RANGE[1])
    grid_w = random.randint(min_w, GRID_SIDE_RANGE[1])
    grid = np.zeros((grid_h, grid_w), dtype=int)
    expected = np.zeros_like(grid)
    blocked = np.zeros_like(grid, dtype=bool)
    blocked[:top_reserved, :] = True
    _place_legend(grid, frame_colors, target_colors)

    order = list(range(frame_count))
    random.shuffle(order)
    for idx in order:
        frame_color = frame_colors[idx]
        target_color = target_colors[idx]
        pattern_color = random.choice([c for c in COLORS if c != frame_color])
        direction = random.choice(DIRS)

        placed = False
        for _ in range(FRAME_PLACEMENT_RETRIES):
            height = random.randint(*FRAME_HEIGHT_RANGE)
            width = random.randint(*FRAME_WIDTH_RANGE)
            if grid_h < top_reserved + height + 1 or grid_w < width + 2:
                continue

            pattern_mask = _random_pattern_mask(height - 2, width - 2, direction)
            if pattern_mask is None:
                continue

            r0 = random.randint(top_reserved, grid_h - height - 1)
            c0 = random.randint(1, grid_w - width - 1)
            r1 = r0 + height - 1
            c1 = c0 + width - 1
            if np.any(blocked[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2]):
                continue

            patch = _make_frame_patch(height, width, frame_color, pattern_color, direction, pattern_mask)
            _paste(grid, patch, r0, c0)

            rows, cols = np.nonzero(pattern_mask)
            points = [(int(r0 + 1 + rr), int(c0 + 1 + cc)) for rr, cc in zip(rows, cols)]
            projected = _project_points(points, (r0, c0, r1, c1), direction)
            if len(set(projected)) < 2:
                grid[r0 : r1 + 1, c0 : c1 + 1] = BG
                continue
            for rr, cc in projected:
                expected[rr, cc] = target_color

            blocked[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True
            placed = True
            break
        if not placed:
            return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
