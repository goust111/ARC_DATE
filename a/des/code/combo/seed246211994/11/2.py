from __future__ import annotations

import random
from collections import deque

import numpy as np


YELLOW = 4
BLUE = 1
RED = 2
GREEN = 3
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(values, minlength=10)
    return int(np.argmax(counts))


def bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def coords_to_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def find_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = np.asarray(grid == color, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))

    components.sort(key=lambda pts: bbox_from_coords(pts)[:2])
    return components


def is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def has_flanked_boundary_gap(mask: np.ndarray) -> bool:
    h, w = mask.shape

    for c in range(w):
        if not mask[0, c] and mask[0, :c].any() and mask[0, c + 1 :].any():
            return True
        if h > 1 and not mask[h - 1, c] and mask[h - 1, :c].any() and mask[h - 1, c + 1 :].any():
            return True

    for r in range(h):
        if not mask[r, 0] and mask[:r, 0].any() and mask[r + 1 :, 0].any():
            return True
        if w > 1 and not mask[r, w - 1] and mask[:r, w - 1].any() and mask[r + 1 :, w - 1].any():
            return True

    return False


def paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    patch = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    patch[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert len(colors) == 1, "expected exactly one foreground color"
    fg = colors[0]

    components = find_components(g, fg)
    assert components, "expected at least one foreground object"

    out = np.full_like(g, bg)
    width = g.shape[1]
    for coords in components:
        bbox = bbox_from_coords(coords)
        r0, _c0, r1, _c1 = bbox
        mask = coords_to_mask(coords, bbox)
        obj_w = mask.shape[1]
        go_right = has_flanked_boundary_gap(mask)
        dst_c = width - obj_w if go_right else 0
        dst = out[r0 : r1 + 1, dst_c : dst_c + obj_w]
        assert np.all(dst[mask] == bg), "objects must not overlap after sliding"
        dst[mask] = GREEN if go_right else RED

    return out


GEN_RETRIES = 400
GRID_H_RANGE = (10, 22)
GRID_W_RANGE = (14, 28)
OBJECT_COUNT_RANGE = (2, 5)
NOTCH_HEIGHT_RANGE = (3, 6)
NOTCH_WIDTH_RANGE = (3, 9)
CLOSED_HEIGHT_RANGE = (2, 6)
CLOSED_WIDTH_RANGE = (2, 9)
MIN_ROW_GAP = 1
INPUT_MARGIN = 1
MAX_FILL_FRACTION = 0.42
MIN_FILL_RATIO = 0.55


def sample_row_starts(heights: list[int], total_h: int) -> list[int]:
    free = total_h - sum(heights) - MIN_ROW_GAP * (len(heights) - 1)
    if free < 0:
        raise ValueError("insufficient height")

    slack = [0] * (len(heights) + 1)
    for _ in range(free):
        slack[random.randrange(len(slack))] += 1

    starts: list[int] = []
    r = slack[0]
    for i, height in enumerate(heights):
        starts.append(r)
        r += height
        if i + 1 < len(heights):
            r += MIN_ROW_GAP + slack[i + 1]
    return starts


def carve_interior(mask: np.ndarray, remaining_budget: int) -> int:
    h, w = mask.shape
    if remaining_budget <= 0 or h < 4 or w < 4:
        return 0

    for _ in range(12):
        r0 = random.randint(1, h - 2)
        r1 = random.randint(r0, h - 2)
        c0 = random.randint(1, w - 2)
        c1 = random.randint(c0, w - 2)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area > remaining_budget:
            continue
        mask[r0 : r1 + 1, c0 : c1 + 1] = False
        return int(area)
    return 0


def make_notch_mask(height: int, width: int) -> np.ndarray | None:
    mask = np.ones((height, width), dtype=bool)
    max_remove = int(mask.size - np.ceil(mask.size * MIN_FILL_RATIO))
    if max_remove < 1:
        return None

    edge = random.choice(["top", "bottom", "left", "right"])
    if edge in {"top", "bottom"}:
        run = random.randint(1, width - 2)
        start = random.randint(1, width - run - 1)
        depth = random.randint(1, min(height - 1, max_remove // run))
        rows = slice(0, depth) if edge == "top" else slice(height - depth, height)
        mask[rows, start : start + run] = False
        removed = run * depth
    else:
        run = random.randint(1, height - 2)
        start = random.randint(1, height - run - 1)
        depth = random.randint(1, min(width - 1, max_remove // run))
        cols = slice(0, depth) if edge == "left" else slice(width - depth, width)
        mask[start : start + run, cols] = False
        removed = run * depth

    if random.random() < 0.45:
        removed += carve_interior(mask, max_remove - removed)

    if not is_connected(mask):
        return None
    if not has_flanked_boundary_gap(mask):
        return None
    return mask


def make_closed_mask(height: int, width: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    max_remove = int(mask.size - np.ceil(mask.size * MIN_FILL_RATIO))

    if max_remove > 0 and random.random() < 0.7:
        removed = carve_interior(mask, max_remove)
        if random.random() < 0.35:
            carve_interior(mask, max_remove - removed)

    assert is_connected(mask)
    assert not has_flanked_boundary_gap(mask)
    return mask


def sample_kinds(count: int) -> list[str]:
    right_count = count // 2 if random.random() < 0.5 else (count + 1) // 2
    right_count = max(1, min(count - 1, right_count))
    kinds = ["right"] * right_count + ["left"] * (count - right_count)
    random.shuffle(kinds)
    return kinds


def sample_dimensions(kinds: list[str]) -> tuple[list[int], list[int]]:
    heights: list[int] = []
    widths: list[int] = []
    for kind in kinds:
        if kind == "right":
            heights.append(random.randint(*NOTCH_HEIGHT_RANGE))
            widths.append(random.randint(*NOTCH_WIDTH_RANGE))
        else:
            heights.append(random.randint(*CLOSED_HEIGHT_RANGE))
            widths.append(random.randint(*CLOSED_WIDTH_RANGE))
    return heights, widths


def render_expected(
    masks: list[np.ndarray], starts: list[int], kinds: list[str], shape: tuple[int, int]
) -> np.ndarray | None:
    h, w = shape
    out = np.full((h, w), YELLOW, dtype=int)
    for mask, r0, kind in zip(masks, starts, kinds):
        dst_c = w - mask.shape[1] if kind == "right" else 0
        patch = out[r0 : r0 + mask.shape[0], dst_c : dst_c + mask.shape[1]]
        if np.any(patch[mask] != YELLOW):
            return None
        patch[mask] = GREEN if kind == "right" else RED
    return out


def render_input(masks: list[np.ndarray], starts: list[int], shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    grid = np.full((h, w), YELLOW, dtype=int)
    for mask, r0 in zip(masks, starts):
        obj_w = mask.shape[1]
        lo = INPUT_MARGIN
        hi = w - obj_w - INPUT_MARGIN
        if hi < lo:
            return None
        c0 = random.randint(lo, hi)
        patch = grid[r0 : r0 + mask.shape[0], c0 : c0 + obj_w]
        if np.any(patch[mask] != YELLOW):
            return None
        patch[mask] = BLUE
    return grid


def gen_attempt() -> np.ndarray | None:
    count = random.randint(*OBJECT_COUNT_RANGE)
    kinds = sample_kinds(count)
    heights, widths = sample_dimensions(kinds)

    min_h = sum(heights) + MIN_ROW_GAP * (count - 1)
    if min_h > GRID_H_RANGE[1]:
        return None
    grid_h = random.randint(max(GRID_H_RANGE[0], min_h), GRID_H_RANGE[1])

    min_w = max(max(widths) + 2 * INPUT_MARGIN + 1, GRID_W_RANGE[0])
    if min_w > GRID_W_RANGE[1]:
        return None
    grid_w = random.randint(min_w, GRID_W_RANGE[1])

    try:
        starts = sample_row_starts(heights, grid_h)
    except ValueError:
        return None

    masks: list[np.ndarray] = []
    for kind, height, width in zip(kinds, heights, widths):
        if kind == "right":
            mask = make_notch_mask(height, width)
            if mask is None:
                return None
        else:
            mask = make_closed_mask(height, width)
        masks.append(mask)

    total_fg = sum(int(mask.sum()) for mask in masks)
    if total_fg > int(grid_h * grid_w * MAX_FILL_FRACTION):
        return None

    expected = render_expected(masks, starts, kinds, (grid_h, grid_w))
    if expected is None:
        return None

    inp = render_input(masks, starts, (grid_h, grid_w))
    if inp is None:
        return None

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
