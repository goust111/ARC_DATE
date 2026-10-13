from __future__ import annotations

from collections import deque

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
BLACK = 0

_RNG = np.random.default_rng()
_MAX_GEN_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("expected a non-empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _axis_from_marker(grid: np.ndarray) -> str:
    red = np.argwhere(grid == RED)
    assert len(red) == 1, "expected exactly one red marker"
    _r, c = map(int, red[0])
    return "vertical" if c <= grid.shape[1] // 2 else "horizontal"


def _mirror_blue_mask(blue: np.ndarray, axis: str) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(blue)
    patch = blue[r0 : r1 + 1, c0 : c1 + 1]
    mirrored = np.zeros_like(blue, dtype=bool)
    if axis == "vertical":
        mirrored[r0 : r1 + 1, c0 : c1 + 1] = np.fliplr(patch)
    else:
        mirrored[r0 : r1 + 1, c0 : c1 + 1] = np.flipud(patch)
    return mirrored


def _flood_exterior(walls: np.ndarray) -> np.ndarray:
    h, w = walls.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def push(r: int, c: int) -> None:
        if 0 <= r < h and 0 <= c < w and not walls[r, c] and not exterior[r, c]:
            exterior[r, c] = True
            q.append((r, c))

    for r in range(h):
        push(r, 0)
        push(r, w - 1)
    for c in range(w):
        push(0, c)
        push(h - 1, c)

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            push(nr, nc)

    return exterior


def _enclosed_mask(walls: np.ndarray) -> np.ndarray:
    return (~walls) & (~_flood_exterior(walls))


def _count_components(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        count += 1
        q: deque[tuple[int, int]] = deque([(r, c)])
        seen[r, c] = True
        while q:
            cr, cc = q.popleft()
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
    return count


def _transform(grid: np.ndarray, forced_axis: str | None = None) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    blue = g == BLUE
    assert blue.any(), "expected a blue sprite"
    axis = forced_axis or _axis_from_marker(g)
    mirrored = _mirror_blue_mask(blue, axis)
    merged = blue | mirrored
    enclosed = _enclosed_mask(merged)

    out = np.full_like(g, BLACK)
    out[merged] = BLUE
    out[enclosed] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the blue sprite across the marker-selected bbox axis, merge it, then fill the new enclosed regions.

    Input structure:
    - A black background contains one 4-connected irregular blue sprite and a single red marker pixel.
    - The red marker sits on the left or right side of the grid and chooses which symmetry axis to use.
    - The mirror axis is taken from the blue sprite's own bounding box, not from the full canvas.

    Rule:
    - If the red marker is in the left half, mirror the blue sprite across the bounding box's vertical mid-axis.
    - If the red marker is in the right half, mirror it across the bounding box's horizontal mid-axis.
    - Take the pixelwise OR of the original sprite and its mirror.
    - Treat the merged blue pixels as walls, flood-fill black background from the border, and recolor any remaining enclosed black cells green.
    - Erase the red marker to black in the output while keeping the merged blue walls unchanged.

    Key insight:
    The green area is not visible in the input sprite alone; it appears only after the marker-selected mirror copy combines with the original to close off space.
    """
    return _transform(grid)


def _build_vertical_half(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(4, 7))
    w = int(rng.integers(5, 8))
    half_w = (w + 1) // 2

    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = True
    mask[0, :half_w] = True
    mask[-1, :half_w] = True

    if rng.random() < 0.45 and h >= 5:
        divider_len = half_w - 1
        if int(mask.sum()) + divider_len <= 15:
            row = int(rng.integers(2, h - 2))
            mask[row, 1:half_w] = True
    else:
        choices = list(range(1, h - 1))
        if choices:
            max_teeth = min(2, 15 - int(mask.sum()))
            tooth_count = int(rng.integers(0, max_teeth + 1))
            if tooth_count:
                rows = rng.choice(choices, size=tooth_count, replace=False)
                mask[np.asarray(rows, dtype=int), 1] = True

    return mask if 8 <= int(mask.sum()) <= 15 else None


def _build_horizontal_half(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(5, 8))
    w = int(rng.integers(4, 7))
    half_h = (h + 1) // 2

    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[:half_h, 0] = True
    mask[:half_h, -1] = True

    if rng.random() < 0.45 and w >= 5:
        divider_len = half_h - 1
        if int(mask.sum()) + divider_len <= 15:
            col = int(rng.integers(2, w - 2))
            mask[1:half_h, col] = True
    else:
        choices = list(range(1, w - 1))
        if choices:
            max_teeth = min(2, 15 - int(mask.sum()))
            tooth_count = int(rng.integers(0, max_teeth + 1))
            if tooth_count:
                cols = rng.choice(choices, size=tooth_count, replace=False)
                mask[1, np.asarray(cols, dtype=int)] = True

    return mask if 8 <= int(mask.sum()) <= 15 else None


def _make_blue_patch(rng: np.random.Generator, axis: str) -> np.ndarray | None:
    builder = _build_vertical_half if axis == "vertical" else _build_horizontal_half
    patch = builder(rng)
    if patch is None:
        return None
    if _count_components(patch) != 1:
        return None
    reflected = np.fliplr(patch) if axis == "vertical" else np.flipud(patch)
    if np.array_equal(patch, reflected):
        return None
    return patch


def _place_patch(grid: np.ndarray, patch: np.ndarray, rng: np.random.Generator) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    row_choices = list(range(2, h - ph - 1))
    col_choices = list(range(2, w - pw - 1))
    if not row_choices or not col_choices:
        return False

    target_r = (h - ph) // 2 + int(rng.integers(-1, 2))
    target_c = (w - pw) // 2 + int(rng.integers(-1, 2))
    target_r = min(max(target_r, row_choices[0]), row_choices[-1])
    target_c = min(max(target_c, col_choices[0]), col_choices[-1])
    grid[target_r : target_r + ph, target_c : target_c + pw][patch] = BLUE
    return True


def _make_candidate(rng: np.random.Generator) -> np.ndarray | None:
    axis = "vertical" if rng.random() < 0.5 else "horizontal"
    patch = _make_blue_patch(rng, axis)
    if patch is None:
        return None

    h = max(int(rng.integers(11, 16)), patch.shape[0] + 4)
    w = max(int(rng.integers(11, 16)), patch.shape[1] + 4)
    grid = np.zeros((h, w), dtype=int)
    if not _place_patch(grid, patch, rng):
        return None

    if axis == "vertical":
        marker = (0, 0) if rng.random() < 0.5 else (h - 1, 0)
    else:
        marker = (0, w - 1) if rng.random() < 0.5 else (h - 1, w - 1)
    grid[marker] = RED
    return grid


def _is_valid_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not set(np.unique(grid)) <= {BLACK, BLUE, RED}:
        return False

    red = np.argwhere(grid == RED)
    if len(red) != 1:
        return False
    corner_set = {
        (0, 0),
        (0, grid.shape[1] - 1),
        (grid.shape[0] - 1, 0),
        (grid.shape[0] - 1, grid.shape[1] - 1),
    }
    if tuple(map(int, red[0])) not in corner_set:
        return False
    blue = grid == BLUE
    if not blue.any():
        return False
    if not (8 <= int(blue.sum()) <= 15):
        return False
    if _count_components(blue) != 1:
        return False

    axis = _axis_from_marker(grid)
    mirrored = _mirror_blue_mask(blue, axis)
    if np.array_equal(mirrored, blue):
        return False

    merged = blue | mirrored
    enclosed = _enclosed_mask(merged)
    if int(enclosed.sum()) == 0:
        return False
    if int(enclosed.sum()) > 15:
        return False
    if _count_components(enclosed) > 2:
        return False
    if _enclosed_mask(blue).any():
        return False
    if _enclosed_mask(mirrored).any():
        return False

    wrong_axis = "horizontal" if axis == "vertical" else "vertical"
    correct_out = _transform(grid, axis)
    wrong_out = _transform(grid, wrong_axis)
    if np.array_equal(correct_out, wrong_out):
        return False
    if not np.array_equal(correct_out == GREEN, enclosed):
        return False
    if not np.array_equal(correct_out == BLUE, merged):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        grid = _make_candidate(_RNG)
        if grid is not None and _is_valid_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid mirrored-enclosure ARC input")
