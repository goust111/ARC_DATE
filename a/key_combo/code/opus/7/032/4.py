from __future__ import annotations

from collections import deque

import numpy as np


BACKGROUND = 0
TOP_COLORS = (1, 3, 7)
SCALE_FACTORS = (2, 3)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((factor, factor), dtype=int)).astype(bool)


def _upper_region_end(grid: np.ndarray) -> int:
    occupied_rows = [r for r in range(grid.shape[0]) if np.any(grid[r] != BACKGROUND)]
    assert occupied_rows, "expected at least one foreground row"
    best_gap = -1
    upper_end = occupied_rows[-1]
    for a, b in zip(occupied_rows, occupied_rows[1:]):
        gap = b - a - 1
        if gap > best_gap:
            best_gap = gap
            upper_end = a
    assert best_gap >= 1, "expected a blank gap between upper and lower regions"
    return upper_end


def _extract_upper_components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BACKGROUND or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            pts = np.array(coords, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[pts[:, 0] - r0, pts[:, 1] - c0] = True
            components.append(
                {
                    "color": color,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                }
            )
    return components


def _matches_scaled_crop(small_mask: np.ndarray, large_mask: np.ndarray) -> bool:
    for factor in SCALE_FACTORS:
        scaled = _scale_mask(small_mask, factor)
        lh, lw = large_mask.shape
        sh, sw = scaled.shape
        if lh > sh or lw > sw:
            continue
        for r0 in range(sh - lh + 1):
            for c0 in range(sw - lw + 1):
                if np.array_equal(scaled[r0 : r0 + lh, c0 : c0 + lw], large_mask):
                    return True
    return False


def _find_activation_color(grid: np.ndarray) -> int:
    upper_end = _upper_region_end(grid)
    upper = grid[: upper_end + 1]
    objects = _extract_upper_components(upper)
    assert len(objects) == 3, "expected exactly three upper objects"

    matches: list[int] = []
    for i, src in enumerate(objects):
        src_mask = np.asarray(src["mask"], dtype=bool)
        for j, dst in enumerate(objects):
            if i == j:
                continue
            dst_mask = np.asarray(dst["mask"], dtype=bool)
            if _matches_scaled_crop(src_mask, dst_mask):
                matches.append(i)

    assert len(matches) == 1, "expected a unique cross-scale pair"
    return int(objects[matches[0]]["color"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the smaller member of the unique cross-scale shape pair to choose which bottom markers trigger upward column filling.

    Input structure:
    - The grid has a black background with a clearly separated upper object area and a lower column-marker area.
    - The upper area contains exactly three 4-connected monochromatic objects colored blue, green, and orange.
    - Two upper objects share the same shape at different integer scales, possibly with the larger visible object clipped by a canvas edge; the bottom row contains sparse color markers and same-color pixels above them.

    Rule:
    - Find the unique upper-object pair where one object's tight mask matches a 2x or 3x scaled crop of the other, ignoring color, and record the smaller object's color.
    - Select only the columns whose bottom-row marker has that color.
    - In each selected column, scan upward from the bottom while carrying the current color: non-background cells refresh the carried color, and background cells are filled with it; all unselected columns remain unchanged.

    Key insight:
    - The shape-matching step does not directly transform geometry; it produces a color key that decides which bottom markers activate the column propagation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    activation_color = _find_activation_color(g)
    out = g.copy()
    h, _w = g.shape
    selected_cols = np.where(g[-1] == activation_color)[0]
    assert selected_cols.size >= 1, "expected at least one activated bottom marker"

    for c in selected_cols:
        current = int(g[-1, c])
        for r in range(h - 1, -1, -1):
            cell = int(g[r, c])
            if cell == BACKGROUND:
                out[r, c] = current
            else:
                current = cell

    return out


def _random_connected_mask(
    rng: np.random.Generator,
    h_range: tuple[int, int],
    w_range: tuple[int, int],
    area_range: tuple[int, int],
) -> np.ndarray:
    for _ in range(200):
        h = int(rng.integers(h_range[0], h_range[1] + 1))
        w = int(rng.integers(w_range[0], w_range[1] + 1))
        max_area = min(area_range[1], h * w - 1)
        min_area = min(area_range[0], max_area)
        if min_area < 1 or max_area < min_area:
            continue
        area = int(rng.integers(min_area, max_area + 1))

        mask = np.zeros((h, w), dtype=bool)
        start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        mask[start] = True
        while int(mask.sum()) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in np.argwhere(mask):
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = frontier[int(rng.integers(0, len(frontier)))]
            mask[nr, nc] = True

        mask = _trim_mask(mask)
        if (
            int(mask.sum()) == area
            and mask.shape[0] >= 2
            and mask.shape[1] >= 2
            and not np.all(mask)
            and _mask_connected(mask)
        ):
            return mask
    raise RuntimeError("failed to build a connected irregular mask")


def _random_scaled_visible_crop(rng: np.random.Generator, small_mask: np.ndarray, factor: int) -> np.ndarray:
    full = _scale_mask(small_mask, factor)
    small_area = int(small_mask.sum())
    for _ in range(300):
        r0 = int(rng.integers(0, full.shape[0]))
        r1 = int(rng.integers(r0 + 1, full.shape[0] + 1))
        c0 = int(rng.integers(0, full.shape[1]))
        c1 = int(rng.integers(c0 + 1, full.shape[1] + 1))
        sub = full[r0:r1, c0:c1]
        if not sub.any():
            continue
        visible = _trim_mask(sub)
        area = int(visible.sum())
        if area < 4 or area > 12 or area <= small_area:
            continue
        if np.array_equal(visible, full):
            continue
        if np.array_equal(visible, small_mask):
            continue
        if visible.shape == small_mask.shape and np.array_equal(visible, small_mask):
            continue
        if visible.shape[0] <= small_mask.shape[0] and visible.shape[1] <= small_mask.shape[1]:
            continue
        if _mask_connected(visible):
            return visible
    raise RuntimeError("failed to crop a visible scaled object")


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = color


def _place_edge_object(
    rng: np.random.Generator,
    canvas_shape: tuple[int, int],
    mask: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int]:
    h, w = mask.shape
    upper_h, width = canvas_shape
    candidates: list[tuple[int, int]] = []
    for c in range(width - w + 1):
        candidates.append((0, c))
    for r in range(upper_h - h + 1):
        candidates.append((r, 0))
        candidates.append((r, width - w))
    order = rng.permutation(len(candidates))
    for idx in order:
        top, left = candidates[int(idx)]
        box = (top, left, top + h - 1, left + w - 1)
        if all(_boxes_separated(box, prev) for prev in boxes):
            return top, left
    raise RuntimeError("failed to place edge object")


def _place_inner_object(
    rng: np.random.Generator,
    canvas_shape: tuple[int, int],
    mask: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int]:
    h, w = mask.shape
    upper_h, width = canvas_shape
    for _ in range(200):
        top = int(rng.integers(0, upper_h - h + 1))
        left = int(rng.integers(0, width - w + 1))
        box = (top, left, top + h - 1, left + w - 1)
        if all(_boxes_separated(box, prev) for prev in boxes):
            return top, left
    raise RuntimeError("failed to place inner object")


def _pick_marker_columns(rng: np.random.Generator, free_cols: list[int], count: int) -> list[int]:
    free_cols = list(sorted(free_cols))
    for min_gap in (2, 1, 0):
        for _ in range(100):
            rng.shuffle(free_cols)
            picked: list[int] = []
            for c in free_cols:
                if all(abs(c - prev) > min_gap for prev in picked):
                    picked.append(int(c))
                    if len(picked) == count:
                        return sorted(picked)
    raise RuntimeError("failed to pick marker columns")


def _build_lower_region(
    rng: np.random.Generator,
    grid: np.ndarray,
    lower_top: int,
    activation_color: int,
    free_cols: list[int],
) -> None:
    h, w = grid.shape
    other_colors = [color for color in TOP_COLORS if color != activation_color]
    counts: dict[int, int] = {activation_color: int(rng.integers(1, 4))}
    if rng.random() < 0.65:
        counts[other_colors[0]] = int(rng.integers(1, 3))
        counts[other_colors[1]] = int(rng.integers(1, 3))
    else:
        chosen = other_colors[int(rng.integers(0, 2))]
        counts[chosen] = int(rng.integers(1, 3))

    total_markers = sum(counts.values())
    marker_cols = _pick_marker_columns(rng, free_cols, total_markers)

    marker_colors: list[int] = []
    for color, count in counts.items():
        marker_colors.extend([color] * count)
    rng.shuffle(marker_colors)

    for col, color in zip(marker_cols, marker_colors):
        grid[h - 1, col] = color
        extra_count = int(rng.integers(1, min(4, h - lower_top)))
        extra_rows = rng.choice(np.arange(lower_top, h - 1), size=extra_count, replace=False)
        for r in extra_rows:
            grid[int(r), col] = color

    for r in range(lower_top, h - 1):
        if not np.any(grid[r] != BACKGROUND):
            col = marker_cols[int(rng.integers(0, len(marker_cols)))]
            grid[r, col] = int(grid[h - 1, col])

    for col in marker_cols:
        if int(grid[h - 1, col]) == activation_color and np.all(grid[lower_top:h, col] != BACKGROUND):
            clear_r = int(rng.integers(lower_top, h - 1))
            grid[clear_r, col] = BACKGROUND


def _generate_candidate(rng: np.random.Generator) -> np.ndarray:
    factor = SCALE_FACTORS[int(rng.integers(0, len(SCALE_FACTORS)))]
    small_mask = _random_connected_mask(rng, (2, 3), (2, 3), (4, 5))
    large_visible_mask = _random_scaled_visible_crop(rng, small_mask, factor)
    distractor_mask = _random_connected_mask(rng, (2, 4), (2, 4), (4, 7))

    min_width = max(13, large_visible_mask.shape[1] + small_mask.shape[1] + distractor_mask.shape[1] + 2)
    width = int(rng.integers(min_width, 21))
    upper_h = int(rng.integers(max(6, large_visible_mask.shape[0]), 10))
    gap = int(rng.integers(2, 5))
    lower_h = int(rng.integers(6, 9))
    height = upper_h + gap + lower_h
    assert height <= 30 and width <= 30

    upper = np.zeros((upper_h, width), dtype=int)
    colors = list(TOP_COLORS)
    rng.shuffle(colors)
    activation_color = colors[0]
    matched_large_color = colors[1]
    distractor_color = colors[2]

    boxes: list[tuple[int, int, int, int]] = []
    top, left = _place_edge_object(rng, upper.shape, large_visible_mask, boxes)
    _paint_mask(upper, large_visible_mask, top, left, matched_large_color)
    boxes.append((top, left, top + large_visible_mask.shape[0] - 1, left + large_visible_mask.shape[1] - 1))

    top, left = _place_inner_object(rng, upper.shape, small_mask, boxes)
    _paint_mask(upper, small_mask, top, left, activation_color)
    boxes.append((top, left, top + small_mask.shape[0] - 1, left + small_mask.shape[1] - 1))

    top, left = _place_inner_object(rng, upper.shape, distractor_mask, boxes)
    _paint_mask(upper, distractor_mask, top, left, distractor_color)

    grid = np.zeros((height, width), dtype=int)
    grid[:upper_h] = upper

    free_cols = [c for c in range(width) if not np.any(upper[:, c] != BACKGROUND)]
    if len(free_cols) < 3:
        raise RuntimeError("not enough free columns for bottom markers")

    lower_top = upper_h + gap
    _build_lower_region(rng, grid, lower_top, activation_color, free_cols)
    return grid


def _valid_generated_grid(grid: np.ndarray) -> bool:
    try:
        activation_color = _find_activation_color(grid)
        bottom_colors = {int(v) for v in grid[-1] if int(v) != BACKGROUND}
        if activation_color not in bottom_colors or len(bottom_colors) < 2:
            return False
        out = main(grid)
    except (AssertionError, RuntimeError, ValueError):
        return False

    if out.shape != grid.shape or np.array_equal(out, grid):
        return False
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        try:
            grid = _generate_candidate(rng)
        except RuntimeError:
            continue
        if _valid_generated_grid(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
