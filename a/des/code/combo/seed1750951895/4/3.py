from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 7
PALETTE = tuple(range(10))
ORTHO_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator bounds.
GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
FRAME_COUNT_RANGE = (2, 5)
FRAME_HEIGHT_RANGE = (3, 8)
FRAME_WIDTH_RANGE = (3, 8)
MAX_PLACEMENT_ATTEMPTS = 80
MAX_GENERATION_ATTEMPTS = 300
EXTERNAL_NOISE_PROB = 0.75
INTERNAL_NOISE_PROB = 0.8
NONFRAME_OBJECT_PROB = 0.4


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ORTHO_DIRS:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _component_bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_rectangular_frame(coords: np.ndarray) -> tuple[bool, int]:
    r0, c0, r1, c1 = _component_bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False, 0

    rel = coords - np.array([r0, c0], dtype=int)
    boundary = (
        (rel[:, 0] == 0)
        | (rel[:, 0] == h - 1)
        | (rel[:, 1] == 0)
        | (rel[:, 1] == w - 1)
    )
    expected_area = 2 * h + 2 * w - 4
    if int(coords.shape[0]) != expected_area or not bool(np.all(boundary)):
        return False, 0
    return True, (h - 2) * (w - 2)


def _frame_objects(grid: np.ndarray) -> list[tuple[int, int]]:
    bg = _most_common_color(grid)
    frames: list[tuple[int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _find_components(grid == color):
            ok, hole_area = _is_rectangular_frame(coords)
            if ok:
                frames.append((hole_area, color))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find hollow rectangular frames, sort them by interior area, and output their colors.

    Input structure:
    - Orange-like background with several non-overlapping monochrome rectangular frames.
    - Each valid frame is one pixel thick and may contain small interior noise objects.
    - Extra non-frame objects may also appear elsewhere.

    Rule:
    - Keep only components that exactly match the outer boundary of their bounding box.
    - Compute each frame's hole area as (height - 2) * (width - 2).
    - Sort frames by decreasing hole area.
    - Output an N x 1 grid containing the frame colors in sorted order.
    """
    g = np.asarray(grid, dtype=int)
    frames = _frame_objects(g)
    frames.sort(key=lambda item: -item[0])
    return np.array([[color] for _, color in frames], dtype=int)


def _sample_unique_hole_areas(count: int) -> list[int]:
    choices = list(range(1, (FRAME_HEIGHT_RANGE[1] - 2) * (FRAME_WIDTH_RANGE[1] - 2) + 1))
    random.shuffle(choices)
    return choices[:count]


def _dims_for_hole_area(area: int) -> list[tuple[int, int]]:
    dims = []
    for inner_h in range(FRAME_HEIGHT_RANGE[0] - 2, FRAME_HEIGHT_RANGE[1] - 1):
        if inner_h <= 0 or area % inner_h != 0:
            continue
        inner_w = area // inner_h
        h = inner_h + 2
        w = inner_w + 2
        if FRAME_HEIGHT_RANGE[0] <= h <= FRAME_HEIGHT_RANGE[1] and FRAME_WIDTH_RANGE[0] <= w <= FRAME_WIDTH_RANGE[1]:
            dims.append((h, w))
    return dims


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int = 1) -> bool:
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(occupied.shape[0], r0 + h + margin)
    cc1 = min(occupied.shape[1], c0 + w + margin)
    return not bool(np.any(occupied[rr0:rr1, cc0:cc1]))


def _place_frames() -> list[tuple[np.ndarray, list[tuple[int, int, int, int]]]] | None:
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    hole_areas = _sample_unique_hole_areas(frame_count)

    candidates = []
    for area in hole_areas:
        dims = _dims_for_hole_area(area)
        if not dims:
            return None
        candidates.append((area, random.choice(dims)))

    random.shuffle(candidates)
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placed: list[tuple[int, int, int, int, int]] = []
    used_colors = {BG}
    frame_colors = random.sample([c for c in PALETTE if c != BG], frame_count)

    for color, (area, (h, w)) in zip(frame_colors, candidates):
        if h > grid_h or w > grid_w:
            return None
        success = False
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, grid_h - h)
            c0 = random.randint(0, grid_w - w)
            if _can_place(occupied, r0, c0, h, w):
                occupied[r0 : r0 + h, c0 : c0 + w] = True
                placed.append((r0, c0, h, w, color))
                used_colors.add(color)
                success = True
                break
        if not success:
            return None
    return [occupied, placed]


def _paint_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color


def _random_shape(max_h: int, max_w: int, area: int) -> np.ndarray:
    h = random.randint(1, max_h)
    w = random.randint(1, max_w)
    h = max(1, h)
    w = max(1, w)
    seed = (random.randint(0, h - 1), random.randint(0, w - 1))
    cells = {seed}
    while len(cells) < min(area, h * w):
        frontier = []
        for r, c in cells:
            for dr, dc in ORTHO_DIRS:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    out = np.zeros((max(rs) - min(rs) + 1, max(cs) - min(cs) + 1), dtype=bool)
    for r, c in cells:
        out[r - min(rs), c - min(cs)] = True
    return out


def _try_place_noise(
    grid: np.ndarray,
    occupied: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    *,
    inside: bool,
) -> bool:
    r0, c0, r1, c1 = bbox
    if inside:
        top = r0 + 1
        left = c0 + 1
        height = r1 - r0 - 1
        width = c1 - c0 - 1
        margin = 0
    else:
        top = 0
        left = 0
        height, width = grid.shape
        margin = 0

    if height <= 0 or width <= 0:
        return False

    shape = _random_shape(min(3, height), min(3, width), random.randint(1, 4))
    sh, sw = shape.shape
    if sh > height or sw > width:
        return False

    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        rr = random.randint(top, top + height - sh)
        cc = random.randint(left, left + width - sw)
        coords = np.argwhere(shape)
        ok = True
        for dr, dc in coords:
            tr = rr + int(dr)
            tc = cc + int(dc)
            if occupied[tr, tc]:
                ok = False
                break
            if inside and not (r0 + 1 <= tr <= r1 - 1 and c0 + 1 <= tc <= c1 - 1):
                ok = False
                break
        if not ok:
            continue
        for dr, dc in coords:
            tr = rr + int(dr)
            tc = cc + int(dc)
            grid[tr, tc] = color
            occupied[tr, tc] = True
        return True
    return False


def _build_input() -> np.ndarray | None:
    placement = _place_frames()
    if placement is None:
        return None
    occupied, placed = placement
    h, w = occupied.shape
    grid = np.full((h, w), BG, dtype=int)

    for r0, c0, fh, fw, color in placed:
        _paint_frame(grid, r0, c0, fh, fw, color)

    noise_colors = [c for c in PALETTE if c != BG]
    frame_bboxes = [(r0, c0, r0 + fh - 1, c0 + fw - 1) for r0, c0, fh, fw, _ in placed]

    for bbox, (_, _, _, _, frame_color) in zip(frame_bboxes, placed):
        if random.random() < INTERNAL_NOISE_PROB:
            options = [c for c in noise_colors if c != frame_color]
            _try_place_noise(grid, occupied, bbox, random.choice(options), inside=True)

    if random.random() < EXTERNAL_NOISE_PROB:
        extra_count = random.randint(1, 4)
        for _ in range(extra_count):
            color = random.choice(noise_colors)
            _try_place_noise(grid, occupied, (0, 0, 0, 0), color, inside=False)

    if random.random() < NONFRAME_OBJECT_PROB:
        for _ in range(random.randint(1, 2)):
            color = random.choice(noise_colors)
            sh = random.randint(1, 4)
            sw = random.randint(1, 4)
            for _ in range(MAX_PLACEMENT_ATTEMPTS):
                if sh > h or sw > w:
                    break
                rr = random.randint(0, h - sh)
                cc = random.randint(0, w - sw)
                if not _can_place(occupied, rr, cc, sh, sw, margin=0):
                    continue
                if sh >= 3 and sw >= 3 and random.random() < 0.5:
                    patch = np.full((sh, sw), BG, dtype=int)
                    patch[:, 0] = color
                    patch[0, :] = color
                else:
                    patch = np.full((sh, sw), color, dtype=int)
                mask = patch != BG
                if not np.any(mask):
                    continue
                grid[rr : rr + sh, cc : cc + sw][mask] = color
                occupied[rr : rr + sh, cc : cc + sw][mask] = True
                break

    frames = _frame_objects(grid)
    expected = sorted(((((fh - 2) * (fw - 2)), color) for _, _, fh, fw, color in placed), key=lambda x: -x[0])
    if sorted(frames, key=lambda x: -x[0]) != expected:
        return None

    bg_count = int(np.count_nonzero(grid == BG))
    if bg_count <= len(grid.ravel()) // 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_input()
        if grid is None:
            continue
        out = main(grid)
        assert out.shape == (len(_frame_objects(grid)), 1)
        return grid
    raise RuntimeError("failed to generate a valid input")
