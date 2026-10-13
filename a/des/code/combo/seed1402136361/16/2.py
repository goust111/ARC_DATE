from __future__ import annotations

import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Fixed output colors from the puzzle statement.
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

# Keep generator input colors disjoint from output colors for visual clarity.
INPUT_PALETTE = (0, 5, 6, 7, 8, 9)
GRID_HEIGHT_RANGE = (15, 22)
GRID_WIDTH_RANGE = (15, 22)
EXTRA_OBJECT_RANGE = (0, 3)
PLACEMENT_TRIES = 80
GENERATION_TRIES = 300
OBJECT_GAP = 1


def _dominant_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=int(grid.max()) + 1)
    return int(np.argmax(counts))


def _background_neighbor_count(grid: np.ndarray, r: int, c: int, bg: int) -> int:
    height, width = grid.shape
    count = 0
    for dr, dc in DIR4:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < height and 0 <= cc < width and int(grid[rr, cc]) == int(bg):
            count += 1
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every foreground pixel by how many of its 4-neighbors are background.

    0 background neighbors -> blue
    1 background neighbor  -> green
    2 background neighbors -> yellow
    3 or 4 background neighbors -> red
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    out = g.copy()
    foreground = np.argwhere(g != bg)

    for r, c in foreground:
        bg_neighbors = _background_neighbor_count(g, int(r), int(c), bg)
        if bg_neighbors == 0:
            out[r, c] = BLUE
        elif bg_neighbors == 1:
            out[r, c] = GREEN
        elif bg_neighbors == 2:
            out[r, c] = YELLOW
        else:
            out[r, c] = RED

    return out


def _mask_from_coords(coords: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    max_r = max(r for r, _ in coords)
    max_c = max(c for _, c in coords)
    mask = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=bool)
    for r, c in coords:
        mask[r - min_r, c - min_c] = True
    return mask


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, dtype=bool, copy=True)
    out = np.rot90(out, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _local_branch_categories(mask: np.ndarray) -> set[int]:
    categories: set[int] = set()
    height, width = mask.shape
    coords = np.argwhere(mask)
    for r, c in coords:
        count = 0
        for dr, dc in DIR4:
            rr = int(r + dr)
            cc = int(c + dc)
            if not (0 <= rr < height and 0 <= cc < width) or not bool(mask[rr, cc]):
                count += 1
        categories.add(min(count, 3))
    return categories


def _solid_rectangle() -> np.ndarray:
    h = random.randint(3, 6)
    w = random.randint(3, 6)
    return np.ones((h, w), dtype=bool)


def _notched_rectangle() -> np.ndarray:
    for _ in range(30):
        h = random.randint(5, 7)
        w = random.randint(5, 7)
        mask = np.ones((h, w), dtype=bool)
        side = random.choice(("top", "bottom", "left", "right"))

        if side in ("top", "bottom"):
            depth = random.randint(1, 2)
            span = random.randint(1, w - 2)
            start = random.randint(1, w - span - 1)
            rows = slice(0, depth) if side == "top" else slice(h - depth, h)
            mask[rows, start : start + span] = False
        else:
            depth = random.randint(1, 2)
            span = random.randint(1, h - 2)
            start = random.randint(1, h - span - 1)
            cols = slice(0, depth) if side == "left" else slice(w - depth, w)
            mask[start : start + span, cols] = False

        if {0, 1, 2}.issubset(_local_branch_categories(mask)):
            return mask
    return _solid_rectangle()


def _armed_rectangle() -> np.ndarray:
    for _ in range(30):
        base_h = random.randint(4, 6)
        base_w = random.randint(4, 6)
        side = random.choice(("top", "bottom", "left", "right"))
        arm_len = random.randint(1, 3)

        if side in ("top", "bottom"):
            span = random.randint(2, min(3, base_w))
            start = random.randint(0, base_w - span)
            mask = np.zeros((base_h + arm_len, base_w), dtype=bool)
            base_top = arm_len if side == "top" else 0
            mask[base_top : base_top + base_h, :] = True
            arm_rows = slice(0, arm_len) if side == "top" else slice(base_h, base_h + arm_len)
            mask[arm_rows, start : start + span] = True
        else:
            span = random.randint(2, min(3, base_h))
            start = random.randint(0, base_h - span)
            mask = np.zeros((base_h, base_w + arm_len), dtype=bool)
            base_left = arm_len if side == "left" else 0
            mask[:, base_left : base_left + base_w] = True
            arm_cols = slice(0, arm_len) if side == "left" else slice(base_w, base_w + arm_len)
            mask[start : start + span, arm_cols] = True

        if {0, 1, 2}.issubset(_local_branch_categories(mask)):
            return mask
    return _solid_rectangle()


def _thick_object() -> np.ndarray:
    builder = random.choice((_solid_rectangle, _notched_rectangle, _armed_rectangle))
    return _random_transform(builder())


def _single_pixel() -> np.ndarray:
    return np.array([[True]], dtype=bool)


def _line() -> np.ndarray:
    length = random.randint(2, 6)
    return np.ones((1, length), dtype=bool)


def _elbow() -> np.ndarray:
    h = random.randint(2, 5)
    w = random.randint(2, 5)
    coords = {(0, c) for c in range(w)}
    coords |= {(r, 0) for r in range(h)}
    return _mask_from_coords(coords)


def _tee() -> np.ndarray:
    bar_w = random.choice((3, 5))
    stem_h = random.randint(2, 4)
    center = bar_w // 2
    coords = {(0, c) for c in range(bar_w)}
    coords |= {(r, center) for r in range(stem_h)}
    return _mask_from_coords(coords)


def _staircase() -> np.ndarray:
    steps = random.randint(2, 4)
    coords = {(0, 0)}
    r = 0
    c = 0
    for _ in range(steps):
        c += 1
        coords.add((r, c))
        r += 1
        coords.add((r, c))
    return _mask_from_coords(coords)


def _thin_object() -> np.ndarray:
    builder = random.choice((_single_pixel, _line, _elbow, _tee, _staircase))
    return _random_transform(builder())


def _mixed_object() -> np.ndarray:
    if random.random() < 0.5:
        return _thick_object()
    return _thin_object()


def _try_place(
    grid: np.ndarray,
    reserved: np.ndarray,
    mask: np.ndarray,
    color: int,
) -> bool:
    height, width = grid.shape
    obj_h, obj_w = mask.shape
    if obj_h + 2 > height or obj_w + 2 > width:
        return False

    max_top = height - obj_h - 1
    max_left = width - obj_w - 1
    if max_top < 1 or max_left < 1:
        return False

    for _ in range(PLACEMENT_TRIES):
        top = random.randint(1, max_top)
        left = random.randint(1, max_left)
        r0 = max(0, top - OBJECT_GAP)
        r1 = min(height, top + obj_h + OBJECT_GAP)
        c0 = max(0, left - OBJECT_GAP)
        c1 = min(width, left + obj_w + OBJECT_GAP)
        if np.any(reserved[r0:r1, c0:c1]):
            continue

        window = grid[top : top + obj_h, left : left + obj_w]
        window[mask] = int(color)
        reserved[r0:r1, c0:c1] = True
        return True

    return False


def _component_sizes(mask: np.ndarray) -> list[int]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes: list[int] = []

    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        size = 0
        while stack:
            r, c = stack.pop()
            size += 1
            for dr, dc in DIR4:
                rr = int(r + dr)
                cc = int(c + dc)
                if (
                    0 <= rr < height
                    and 0 <= cc < width
                    and bool(mask[rr, cc])
                    and not bool(seen[rr, cc])
                ):
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        sizes.append(size)

    return sizes


def _grid_branch_categories(grid: np.ndarray, bg: int) -> set[int]:
    categories: set[int] = set()
    for r, c in np.argwhere(grid != bg):
        categories.add(min(_background_neighbor_count(grid, int(r), int(c), bg), 3))
    return categories


def _valid_input(grid: np.ndarray, bg: int, fg: int, expected_objects: int) -> bool:
    if int(_dominant_color(grid)) != int(bg):
        return False
    if int(np.count_nonzero(grid == bg)) <= int(np.count_nonzero(grid == fg)):
        return False
    if set(np.unique(grid).tolist()) != {int(bg), int(fg)}:
        return False

    components = _component_sizes(grid == fg)
    if len(components) != expected_objects:
        return False
    if min(components) < 1:
        return False

    categories = _grid_branch_categories(grid, bg)
    if categories != {0, 1, 2, 3}:
        return False

    out = main(grid)
    needed_output_colors = {BLUE, GREEN, YELLOW, RED}
    if not needed_output_colors.issubset(set(np.unique(out).tolist())):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    bg, fg = random.sample(INPUT_PALETTE, 2)

    grid = np.full((height, width), int(bg), dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    masks = [_thick_object(), _thin_object()]
    for _ in range(random.randint(*EXTRA_OBJECT_RANGE)):
        masks.append(_mixed_object())
    random.shuffle(masks)
    masks.sort(key=lambda m: int(np.count_nonzero(m)), reverse=True)

    for mask in masks:
        if not _try_place(grid, reserved, mask, fg):
            return None

    if not _valid_input(grid, bg, fg, len(masks)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_TRIES} attempts")
