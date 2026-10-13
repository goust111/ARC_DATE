from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
BLOCKER = 5
OBJECT_COLORS = (BLUE, RED)
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _swap_colors(values: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    out = values.copy()
    mask_a = values == int(color_a)
    mask_b = values == int(color_b)
    out[mask_a] = int(color_b)
    out[mask_b] = int(color_a)
    return out


def _object_info(grid: np.ndarray, bg: int) -> dict:
    g = np.asarray(grid, dtype=int)
    object_mask = (g != int(bg)) & (g != BLOCKER)
    comps = _components(object_mask)
    assert len(comps) == 1, "expected exactly one non-blocker foreground object"
    coords = comps[0]
    r0, c0, r1, c1 = _bbox(coords)
    height = int(r1 - r0 + 1)
    assert height % 2 == 0, "object bbox height must be even"

    colors = sorted(set(map(int, g[object_mask].ravel())))
    assert colors == [BLUE, RED], "object must use blue and red"

    split_row = int(r0 + height // 2)
    upper_coords = [(r, c) for r, c in coords if r < split_row]
    lower_coords = [(r, c) for r, c in coords if r >= split_row]
    assert upper_coords and lower_coords, "both halves must be non-empty"

    upper_boundary_cols = sorted(c for r, c in upper_coords if r == split_row - 1)
    lower_boundary_cols = sorted(c for r, c in lower_coords if r == split_row)
    assert len(upper_boundary_cols) >= 2, "upper gap-edge row needs at least two pixels"
    assert len(lower_boundary_cols) >= 2, "lower gap-edge row needs at least two pixels"

    shift = int(max(lower_boundary_cols) + 1 - min(upper_boundary_cols))
    assert shift > 0, "upper half must shift right"
    return {
        "object_mask": object_mask,
        "bbox": (r0, c0, r1, c1),
        "split_row": split_row,
        "shift": shift,
        "upper_coords": upper_coords,
        "lower_coords": lower_coords,
        "colors": (BLUE, RED),
    }


def _transform_base(grid: np.ndarray, *, swap: bool = True, slide: bool = True) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    info = _object_info(g, bg)
    color_a, color_b = info["colors"]
    shift = info["shift"] if slide else 0

    out = np.full_like(g, int(bg))
    out[g == BLOCKER] = BLOCKER

    lower_rows = np.array([r for r, _ in info["lower_coords"]], dtype=int)
    lower_cols = np.array([c for _, c in info["lower_coords"]], dtype=int)
    upper_rows = np.array([r for r, _ in info["upper_coords"]], dtype=int)
    upper_cols = np.array([c for _, c in info["upper_coords"]], dtype=int)
    assert upper_cols.size > 0 and lower_cols.size > 0
    assert int(upper_cols.max()) + shift < g.shape[1], "shifted upper half exceeds grid width"

    lower_vals = g[lower_rows, lower_cols]
    upper_vals = g[upper_rows, upper_cols]
    if swap:
        lower_vals = _swap_colors(lower_vals, color_a, color_b)
        upper_vals = _swap_colors(upper_vals, color_a, color_b)

    out[lower_rows, lower_cols] = lower_vals
    out[upper_rows, upper_cols + shift] = upper_vals
    return out, info


def _emitters(base: np.ndarray, split_row: int, bg: int) -> list[tuple[int, int, int]]:
    emitters: list[tuple[int, int, int]] = []
    if split_row - 1 >= 0:
        for c in range(base.shape[1]):
            val = int(base[split_row - 1, c])
            if val not in (int(bg), BLOCKER):
                emitters.append((split_row - 1, c, 1))
    if split_row < base.shape[0]:
        for c in range(base.shape[1]):
            val = int(base[split_row, c])
            if val not in (int(bg), BLOCKER):
                emitters.append((split_row, c, -1))
    emitters.sort()
    return emitters


def _ray_path(base: np.ndarray, r: int, c: int, dc: int, bg: int) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    cc = c + dc
    while 0 <= cc < base.shape[1] and int(base[r, cc]) == int(bg):
        path.append((r, cc))
        cc += dc
    return path


def _paint_rays(base: np.ndarray, split_row: int, bg: int) -> tuple[np.ndarray, list[dict]]:
    out = base.copy()
    details: list[dict] = []
    for r, c, dc in _emitters(base, split_row, bg):
        path = _ray_path(base, r, c, dc, bg)
        color = int(base[r, c])
        for rr, cc in path:
            out[rr, cc] = color
        details.append({"start": (r, c), "dir": dc, "color": color, "path": path})
    return out, details


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the two-color object, swap its colors, slide the upper half right, then emit blocker-clipped horizontal rays from the exposed gap edges.

    Input structure:
    - The grid is mostly black background, with one 4-connected blue/red object whose bounding box has even height.
    - The object has foreground pixels on both rows adjacent to its horizontal midpoint, and grey single-pixel blockers sit away from the object.
    - The upper and lower halves are defined by splitting the object's bbox at its horizontal midline.

    Rule:
    - Split the object into upper and lower halves at the bbox midpoint, then swap blue and red in both halves.
    - Keep the lower half fixed and slide the upper half right until its bottom-row leftmost pixel is diagonally adjacent to the lower half's top-row rightmost pixel.
    - After this slide, treat every lower-half top-row pixel as a leftward ray source and every upper-half bottom-row pixel as a rightward ray source.
    - Each ray uses its source pixel's swapped color and extends over background cells only, stopping before a grey blocker, any transformed foreground pixel, or the grid boundary; later ray sources in raster order overwrite earlier ray paint on overlaps.

    Key insight:
    - The ray sources are not marked in the input: they are created only after the split, recolor, and slide expose the two gap-edge rows, so geometry and color-swap jointly determine where rays start and what color they use.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    base, info = _transform_base(g, swap=True, slide=True)
    bg = _most_common_color(g)
    out, _ = _paint_rays(base, info["split_row"], bg)
    return out


def _row_with_optional_hole(start: int, length: int, force_hole: bool) -> set[int]:
    cols = set(range(start, start + length))
    if force_hole and length >= 4:
        hole = random.randint(start + 1, start + length - 2)
        cols.remove(hole)
    return cols


def _grow_half(boundary_row: set[int], half_h: int, width: int) -> list[set[int]]:
    rows = [set(boundary_row)]
    prev = set(boundary_row)
    allowed_min = 4
    allowed_max = width - 5
    for depth in range(1, half_h):
        prev_min, prev_max = min(prev), max(prev)
        span = prev_max - prev_min + 1
        min_w = max(2, span - 2)
        max_w = min(8, span + 2, allowed_max - allowed_min + 1)
        if depth == 1 and len(prev) < span:
            min_w = max(min_w, span)
        if min_w > max_w:
            return []
        row_w = random.randint(min_w, max_w)
        max_left = allowed_max - row_w + 1
        left_lo = max(allowed_min, prev_min - 2)
        left_hi = min(max_left, prev_max)
        candidates = [left for left in range(left_lo, left_hi + 1) if left + row_w - 1 >= prev_min]
        if not candidates:
            candidates = [
                left
                for left in range(allowed_min, max_left + 1)
                if left <= prev_max and left + row_w - 1 >= prev_min
            ]
        if not candidates:
            return []
        left = random.choice(candidates)
        prev = set(range(left, left + row_w))
        rows.append(prev)
    return rows


def _mask_from_rows(rows: list[set[int]], start_row: int, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for offset, cols in enumerate(rows):
        rr = start_row + offset
        for cc in cols:
            mask[rr, cc] = True
    return mask


def _connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _colorize_object(object_mask: np.ndarray, split_row: int) -> np.ndarray:
    coords = np.argwhere(object_mask)
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    c0 = int(cols.min())
    phase = random.randint(0, 1)

    grid = np.full(object_mask.shape, BG, dtype=int)
    for r, c in coords:
        val = BLUE if ((int(r) - r0 + int(c) - c0 + phase) % 2 == 0) else RED
        grid[int(r), int(c)] = val

    for rr in (split_row - 1, split_row):
        boundary_cols = [c for c in range(grid.shape[1]) if grid[rr, c] in OBJECT_COLORS]
        if len(boundary_cols) < 2:
            continue
        colors = {int(grid[rr, c]) for c in boundary_cols}
        if len(colors) == 1:
            grid[rr, boundary_cols[-1]] = BLUE if int(grid[rr, boundary_cols[-1]]) == RED else RED
    return grid


def _non_adjacent_candidates(input_object_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = input_object_mask.shape
    bad = input_object_mask.copy()
    for r, c in np.argwhere(input_object_mask):
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w:
                bad[nr, nc] = True
    return [(r, c) for r in range(h) for c in range(w) if not bad[r, c]]


def _safe_add_blocker(
    blockers: set[tuple[int, int]],
    pos: tuple[int, int],
    allowed: set[tuple[int, int]],
) -> bool:
    if pos not in allowed:
        return False
    r, c = pos
    for rr, cc in blockers:
        if abs(rr - r) + abs(cc - c) <= 1:
            return False
    blockers.add(pos)
    return True


def _render_input(shape: tuple[int, int], object_grid: np.ndarray, blockers: set[tuple[int, int]]) -> np.ndarray:
    grid = object_grid.copy()
    for r, c in blockers:
        grid[r, c] = BLOCKER
    return grid


def _manual_output(base: np.ndarray, split_row: int) -> tuple[np.ndarray, list[dict]]:
    return _paint_rays(base, split_row, BG)


def _build_candidate() -> tuple[np.ndarray, np.ndarray, dict] | None:
    h = random.randint(14, 18)
    w = random.randint(18, 24)
    half_h = random.randint(2, 4)
    top = random.randint(1, h - 2 * half_h - 1)
    split_row = top + half_h

    max_total = w - 8
    lower_len = random.randint(3, min(6, max_total - 3))
    upper_len = random.randint(3, min(6, max_total - lower_len))
    left_max = w - 5 - (lower_len + upper_len)
    if left_max < 4:
        return None
    lower_left = random.randint(4, left_max)
    upper_left = lower_left + lower_len

    lower_hole = lower_len >= 4 and random.random() < 0.8
    upper_hole = upper_len >= 4 and random.random() < 0.8
    if not lower_hole and not upper_hole:
        if random.random() < 0.5 and lower_len >= 4:
            lower_hole = True
        elif upper_len >= 4:
            upper_hole = True

    lower_boundary = _row_with_optional_hole(lower_left, lower_len, lower_hole)
    upper_boundary = _row_with_optional_hole(upper_left, upper_len, upper_hole)
    lower_rows = _grow_half(lower_boundary, half_h, w)
    upper_rows_from_bottom = _grow_half(upper_boundary, half_h, w)
    if not lower_rows or not upper_rows_from_bottom:
        return None
    upper_rows = list(reversed(upper_rows_from_bottom))

    transformed_upper_mask = _mask_from_rows(upper_rows, top, (h, w))
    transformed_lower_mask = _mask_from_rows(lower_rows, split_row, (h, w))
    if np.any(transformed_upper_mask & transformed_lower_mask):
        return None

    upper_min_t = int(np.argwhere(transformed_upper_mask)[:, 1].min())
    upper_max_t = int(np.argwhere(transformed_upper_mask)[:, 1].max())
    lower_min = int(np.argwhere(transformed_lower_mask)[:, 1].min())
    lower_max = int(np.argwhere(transformed_lower_mask)[:, 1].max())
    if lower_min < 4 or upper_max_t > w - 5:
        return None

    shift_cap = min(int(max(upper_boundary) - lower_min), upper_min_t - 4, 6)
    if shift_cap < 1:
        return None
    shift = random.randint(2, shift_cap) if shift_cap >= 2 else 1

    input_upper_mask = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(transformed_upper_mask):
        input_upper_mask[int(r), int(c) - shift] = True
    input_lower_mask = transformed_lower_mask.copy()
    input_object_mask = input_upper_mask | input_lower_mask
    if not _connected(input_object_mask):
        return None

    comps = _components(input_object_mask)
    if len(comps) != 1:
        return None
    r0, c0, r1, c1 = _bbox(comps[0])
    if c0 < 4 or c1 > w - 5:
        return None
    if r1 - r0 + 1 != 2 * half_h:
        return None

    input_object_grid = _colorize_object(input_object_mask, split_row)
    transformed_no_blockers = np.full((h, w), BG, dtype=int)
    lower_coords = np.argwhere(input_lower_mask)
    upper_coords = np.argwhere(input_upper_mask)
    lower_vals = _swap_colors(input_object_grid[lower_coords[:, 0], lower_coords[:, 1]], BLUE, RED)
    upper_vals = _swap_colors(input_object_grid[upper_coords[:, 0], upper_coords[:, 1]], BLUE, RED)
    transformed_no_blockers[lower_coords[:, 0], lower_coords[:, 1]] = lower_vals
    transformed_no_blockers[upper_coords[:, 0], upper_coords[:, 1] + shift] = upper_vals

    emitters = _emitters(transformed_no_blockers, split_row, BG)
    if not emitters:
        return None

    allowed = set(_non_adjacent_candidates(input_object_mask))
    pathful = []
    for r, c, dc in emitters:
        path = _ray_path(transformed_no_blockers, r, c, dc, BG)
        pathful.append({"start": (r, c), "dir": dc, "path": path})
    if sum(len(item["path"]) > 0 for item in pathful) < 2:
        return None
    if not any(item["dir"] == -1 and item["path"] for item in pathful):
        return None
    if not any(item["dir"] == 1 and item["path"] for item in pathful):
        return None

    blockers: set[tuple[int, int]] = set()
    targetable = []
    for item in pathful:
        good_cells = [cell for idx, cell in enumerate(item["path"]) if idx >= 1 and cell in allowed]
        if good_cells:
            targetable.append((item, good_cells))
    if not targetable:
        return None
    random.shuffle(targetable)
    for item, good_cells in targetable[: random.randint(1, min(3, len(targetable)))]:
        _safe_add_blocker(blockers, random.choice(good_cells), allowed)

    candidate_pool = list(allowed - blockers)
    random.shuffle(candidate_pool)
    target_count = random.randint(4, 8)
    for pos in candidate_pool:
        if len(blockers) >= target_count:
            break
        _safe_add_blocker(blockers, pos, allowed)
    if len(blockers) < 3:
        return None

    inp = _render_input((h, w), input_object_grid, blockers)
    transformed_with_blockers = transformed_no_blockers.copy()
    for r, c in blockers:
        transformed_with_blockers[r, c] = BLOCKER
    expected, ray_details = _manual_output(transformed_with_blockers, split_row)
    if sum(len(item["path"]) for item in ray_details) == 0:
        return None

    no_blockers_output, _ = _manual_output(transformed_no_blockers, split_row)

    no_swap_base, _ = _transform_base(inp, swap=False, slide=True)
    no_swap_output, _ = _manual_output(no_swap_base, split_row)

    no_slide_base, _ = _transform_base(inp, swap=True, slide=False)
    no_slide_output, _ = _manual_output(no_slide_base, split_row)

    info = {
        "split_row": split_row,
        "shift": shift,
        "blockers": blockers,
        "ray_details": ray_details,
        "no_blockers_output": no_blockers_output,
        "no_swap_output": no_swap_output,
        "no_slide_output": no_slide_output,
    }
    return inp, expected, info


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, info: dict) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    bg = _most_common_color(inp)
    obj = _object_info(inp, bg)
    coords = np.argwhere(obj["object_mask"])
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    if c0 < 4 or c1 > inp.shape[1] - 5:
        return False

    if np.array_equal(expected, info["no_blockers_output"]):
        return False
    if np.array_equal(expected, info["no_swap_output"]):
        return False
    if np.array_equal(expected, info["no_slide_output"]):
        return False

    rays = info["ray_details"]
    if not any(item["dir"] == -1 and len(item["path"]) > 0 for item in rays):
        return False
    if not any(item["dir"] == 1 and len(item["path"]) > 0 for item in rays):
        return False
    if sum(len(item["path"]) > 0 for item in rays) < 2:
        return False

    blocker_mask = inp == BLOCKER
    if int(blocker_mask.sum()) < 3:
        return False
    object_mask = obj["object_mask"]
    for r, c in np.argwhere(blocker_mask):
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < inp.shape[0] and 0 <= nc < inp.shape[1] and object_mask[nr, nc]:
                return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_candidate()
        if built is None:
            continue
        inp, expected, info = built
        if _validate_semantics(inp, expected, info):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
