from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_RETRIES = 500
PLACEMENT_RETRIES = 150

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[dict[str, object]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict[str, object]] = []

    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )

    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert pts.size, "expected a non-empty mask"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter right-panel objects by target rows marked in the left panel and recolor their target-row pixels.

    Input structure:
    - The background is Black.
    - A full-height Grey vertical separator splits the grid into a left marker panel and a right object panel.
    - Each left-panel row contains at most one non-Black marker pixel; marker rows define the target rows and marker colors.
    - The right panel contains separated 4-connected foreground objects on the Black background.

    Rule:
    - Read every left-panel marker row and its color.
    - For each 4-connected foreground object in the right panel, keep it only when its bounding box intersects at least one target row.
    - Erase every object whose bounding box misses all target rows.
    - In each kept object, recolor pixels lying on a target row to that row's left-panel marker color.
    - Remove the left panel and Grey separator, returning only the transformed right panel.

    Key insight:
    - The left panel is a row selector and color legend: marked rows decide which right-panel objects survive and which row colors are injected.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h > 0 and w >= 3, "expected non-empty left and right panels"

    separator_cols = np.flatnonzero(np.all(g == GREY, axis=0))
    assert separator_cols.size == 1, "expected exactly one full-height Grey divider"
    sep = int(separator_cols[0])
    assert 0 < sep < w - 1, "divider must have a non-empty panel on both sides"

    left = g[:, :sep]
    right = g[:, sep + 1 :]
    target_colors: dict[int, int] = {}
    for r in range(h):
        marker_cols = np.flatnonzero(left[r] != BLACK)
        assert marker_cols.size <= 1, "each left-panel row may have at most one marker"
        if marker_cols.size:
            color = int(left[r, int(marker_cols[0])])
            assert color != BLACK, "target marker must be foreground"
            target_colors[int(r)] = color
    assert target_colors, "expected at least one target row marker"

    target_rows = set(target_colors)
    out = np.full_like(right, BLACK)
    for comp in _components(right != BLACK):
        coords = comp["coords"]
        r0, _c0, r1, _c1 = comp["bbox"]
        if not any(int(r0) <= row <= int(r1) for row in target_rows):
            continue

        coords_arr = np.asarray(coords, dtype=int)
        out[coords_arr[:, 0], coords_arr[:, 1]] = right[coords_arr[:, 0], coords_arr[:, 1]]
        for row, color in target_colors.items():
            if int(r0) <= row <= int(r1):
                row_mask = coords_arr[:, 0] == row
                out[coords_arr[row_mask, 0], coords_arr[row_mask, 1]] = int(color)

    return out


def _make_connected_mask() -> np.ndarray:
    rows = random.randint(2, 5)
    cols = random.randint(2, 5)
    max_area = min(rows * cols, 9)
    area = random.randint(2, max_area)

    mask = np.zeros((rows, cols), dtype=bool)
    start = (random.randrange(rows), random.randrange(cols))
    mask[start] = True
    while int(mask.sum()) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for dr, dc in _NEIGHBORS4:
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < rows and 0 <= nc < cols and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break
        mask[random.choice(tuple(frontier))] = True

    return _crop_mask(mask)


def _valid_top_rows(mask: np.ndarray, target_rows: set[int], keep: bool, height: int) -> list[int]:
    mh = int(mask.shape[0])
    rows: list[int] = []
    for r0 in range(0, height - mh + 1):
        bbox_hits = any(r0 <= row < r0 + mh for row in target_rows)
        if bbox_hits == keep:
            rows.append(r0)
    return rows


def _place_object(
    panel: np.ndarray,
    occupied: np.ndarray,
    mask: np.ndarray,
    color: int,
    target_rows: set[int],
    keep: bool,
) -> bool:
    h, w = panel.shape
    mh, mw = mask.shape
    if mh > h or mw > w:
        return False

    top_rows = _valid_top_rows(mask, target_rows, keep, h)
    if not top_rows:
        return False

    blocked = _dilate(occupied)
    candidates: list[tuple[int, int]] = []
    for r0 in top_rows:
        for c0 in range(0, w - mw + 1):
            if not bool((blocked[r0 : r0 + mh, c0 : c0 + mw] & mask).any()):
                candidates.append((r0, c0))
    if not candidates:
        return False

    r0, c0 = random.choice(candidates)
    panel[r0 : r0 + mh, c0 : c0 + mw][mask] = int(color)
    occupied[r0 : r0 + mh, c0 : c0 + mw][mask] = True
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(10, 24)
    left_w = random.randint(2, 7)
    right_w = random.randint(8, min(20, 30 - left_w - 1))

    marker_count = random.randint(2, min(4, max(2, height // 4)))
    object_count = random.randint(3, min(6, len(PALETTE) - marker_count))

    colors = list(PALETTE)
    random.shuffle(colors)
    marker_colors = colors[:marker_count]
    object_colors = colors[marker_count : marker_count + object_count]

    target_rows_list = random.sample(range(height), marker_count)
    target_rows = set(map(int, target_rows_list))

    grid = np.full((height, left_w + 1 + right_w), BLACK, dtype=int)
    grid[:, left_w] = GREY
    for row, color in zip(target_rows_list, marker_colors):
        grid[int(row), random.randrange(left_w)] = int(color)

    right = np.full((height, right_w), BLACK, dtype=int)
    occupied = np.zeros_like(right, dtype=bool)

    keep_count = object_count // 2
    if object_count % 2 and random.random() < 0.5:
        keep_count += 1
    keep_count = max(1, min(object_count - 1, keep_count))
    keep_flags = [True] * keep_count + [False] * (object_count - keep_count)
    random.shuffle(keep_flags)

    for color, keep in zip(object_colors, keep_flags):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            mask = _make_connected_mask()
            if _place_object(right, occupied, mask, int(color), target_rows, bool(keep)):
                placed = True
                break
        if not placed:
            return None

    grid[:, left_w + 1 :] = right

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != right.shape:
        return None
    if len(_components(right != BLACK)) != object_count:
        return None
    erased_pixels = (right != BLACK) & (out == BLACK)
    recolored_pixels = (right != BLACK) & (out != BLACK) & (out != right)
    if not bool(erased_pixels.any()) or not bool(recolored_pixels.any()):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid row-selector input")
