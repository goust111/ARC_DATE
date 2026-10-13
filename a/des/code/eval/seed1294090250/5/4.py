import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = tuple(range(10))
PERIOD_CHOICES = (3, 4, 5)
WINDOW_CELL_COUNT_MIN = 4
WINDOW_CELL_COUNT_MAX = 6
PATCH_COUNT_MIN = 3
PATCH_COUNT_MAX = 5
MARGIN_MIN = 1
MARGIN_MAX = 3
WINDOW_GAP_MIN = 2
WINDOW_GAP_MAX = 4
GEN_RETRIES = 300


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _largest_monochrome_component(grid: np.ndarray, bg: int) -> dict:
    best = None
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components4(grid == color):
            item = {
                "color": color,
                "coords": coords,
                "area": int(coords.shape[0]),
                "bbox": _bbox(coords),
            }
            if best is None or item["area"] > best["area"]:
                best = item
    assert best is not None, "expected at least one foreground component"
    return best


def _find_hole_bbox(window: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    h, w = window.shape
    best = None
    for coords in _components4(window == bg):
        r0, c0, r1, c1 = _bbox(coords)
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        if coords.shape[0] != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        item = {
            "bbox": (r0, c0, r1, c1),
            "area": int(coords.shape[0]),
        }
        if best is None or item["area"] > best["area"]:
            best = item
    assert best is not None, "expected one internal rectangular hole"
    return best["bbox"]


def _infer_full_window_mask(
    window: np.ndarray, hole_bbox: tuple[int, int, int, int], window_color: int
) -> np.ndarray:
    h, w = window.shape
    row_counts = np.sum(window == window_color, axis=1)
    col_counts = np.sum(window == window_color, axis=0)
    line_rows = row_counts >= (w + 1) // 2
    line_cols = col_counts >= (h + 1) // 2
    assert line_rows[0] and line_rows[-1], "window needs a top/bottom frame"
    assert line_cols[0] and line_cols[-1], "window needs a left/right frame"

    full = line_rows[:, None] | line_cols[None, :]
    hole_mask = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = hole_bbox
    hole_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    observed = window == window_color
    assert np.array_equal(observed[~hole_mask], full[~hole_mask]), "window pattern must be regular"
    return full


def _find_library(
    grid: np.ndarray,
    bg: int,
    window_color: int,
    hole_h: int,
    hole_w: int,
    window_bbox: tuple[int, int, int, int],
) -> dict:
    wr0, wc0, wr1, wc1 = window_bbox
    del wr0, wr1, wc1
    for color in np.unique(grid):
        color = int(color)
        if color in {bg, window_color}:
            continue
        for coords in _components4(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            comp_h = r1 - r0 + 1
            comp_w = c1 - c0 + 1
            if c0 <= wc0:
                continue
            if comp_h != hole_h + 2:
                continue
            unit = hole_w + 1
            if (comp_w - 1) % unit != 0:
                continue
            patch_count = (comp_w - 1) // unit
            if patch_count < 3:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            sep_cols = [idx * unit for idx in range(patch_count + 1)]
            if not np.all(sub[0, :] == color) or not np.all(sub[-1, :] == color):
                continue
            if not all(np.all(sub[:, col] == color) for col in sep_cols):
                continue
            patches = []
            ok = True
            for idx in range(patch_count):
                left = sep_cols[idx] + 1
                right = sep_cols[idx + 1]
                slot = sub[1:-1, left:right]
                if slot.shape != (hole_h, hole_w):
                    ok = False
                    break
                if np.any(slot == color):
                    ok = False
                    break
                patches.append(slot.copy())
            if ok:
                return {
                    "sep_color": color,
                    "bbox": (r0, c0, r1, c1),
                    "patches": patches,
                }
    raise AssertionError("candidate library not found")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the broken window by selecting the candidate patch whose border pattern
    continues the regular lattice hidden by the hole.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    window_comp = _largest_monochrome_component(g, bg)
    window_color = int(window_comp["color"])
    wr0, wc0, wr1, wc1 = window_comp["bbox"]
    window = g[wr0 : wr1 + 1, wc0 : wc1 + 1]

    hole_r0, hole_c0, hole_r1, hole_c1 = _find_hole_bbox(window, bg)
    hole_h = hole_r1 - hole_r0 + 1
    hole_w = hole_c1 - hole_c0 + 1
    full_mask = _infer_full_window_mask(window, (hole_r0, hole_c0, hole_r1, hole_c1), window_color)
    expected_patch = full_mask[hole_r0 : hole_r1 + 1, hole_c0 : hole_c1 + 1]

    library = _find_library(g, bg, window_color, hole_h, hole_w, window_comp["bbox"])
    patch_masks = [(patch != bg) for patch in library["patches"]]
    matches = [idx for idx, patch in enumerate(patch_masks) if np.array_equal(patch, expected_patch)]
    assert len(matches) == 1, "exactly one candidate patch must fit the window"

    out = g.copy()
    lr0, lc0, lr1, lc1 = library["bbox"]
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = bg
    hole_view = out[wr0 + hole_r0 : wr0 + hole_r1 + 1, wc0 + hole_c0 : wc0 + hole_c1 + 1]
    hole_view[:] = bg
    hole_view[expected_patch] = window_color
    return out


def _phase_mask(height: int, width: int, period: int, row_phase: int, col_phase: int) -> np.ndarray:
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    return (rows % period == row_phase) | (cols % period == col_phase)


def _window_mask(height: int, width: int, period: int) -> np.ndarray:
    return _phase_mask(height, width, period, 0, 0)


def _make_window(
    height: int,
    width: int,
    period: int,
    hole_top: int,
    hole_left: int,
    hole_h: int,
    hole_w: int,
    bg: int,
    window_color: int,
) -> tuple[np.ndarray, np.ndarray]:
    base_mask = _window_mask(height, width, period)
    window = np.full((height, width), bg, dtype=int)
    window[base_mask] = window_color
    window[hole_top : hole_top + hole_h, hole_left : hole_left + hole_w] = bg
    return window, base_mask[hole_top : hole_top + hole_h, hole_left : hole_left + hole_w]


def _hole_positions(size: int, hole_size: int, period: int) -> list[int]:
    positions = []
    for start in range(1, size - hole_size):
        end = start + hole_size - 1
        if start % period == 0 or end % period == 0:
            continue
        if not any(idx % period == 0 for idx in range(start, start + hole_size)):
            continue
        positions.append(start)
    return positions


def _unique_phase_masks(height: int, width: int, period: int) -> list[np.ndarray]:
    seen = {}
    for row_phase in range(period):
        for col_phase in range(period):
            mask = _phase_mask(height, width, period, row_phase, col_phase)
            seen.setdefault(mask.tobytes(), mask)
    return list(seen.values())


def _build_candidate_library(
    masks: list[np.ndarray], colors: list[int], bg: int, sep_color: int
) -> np.ndarray:
    patch_h, patch_w = masks[0].shape
    count = len(masks)
    library = np.full((patch_h + 2, count * patch_w + count + 1), sep_color, dtype=int)
    for idx, (mask, color) in enumerate(zip(masks, colors)):
        left = 1 + idx * (patch_w + 1)
        slot = library[1 : patch_h + 1, left : left + patch_w]
        slot[:] = bg
        slot[mask] = color
    return library


def _expected_output(
    inp: np.ndarray,
    window_top: int,
    window_left: int,
    hole_top: int,
    hole_left: int,
    expected_patch: np.ndarray,
    window_color: int,
    library_top: int,
    library_left: int,
    library_shape: tuple[int, int],
    bg: int,
) -> np.ndarray:
    out = inp.copy()
    lib_h, lib_w = library_shape
    out[library_top : library_top + lib_h, library_left : library_left + lib_w] = bg
    hole = out[
        window_top + hole_top : window_top + hole_top + expected_patch.shape[0],
        window_left + hole_left : window_left + hole_left + expected_patch.shape[1],
    ]
    hole[:] = bg
    hole[expected_patch] = window_color
    return out


def _gen_attempt() -> np.ndarray | None:
    period = random.choice(PERIOD_CHOICES)
    cell_rows = random.randint(WINDOW_CELL_COUNT_MIN, WINDOW_CELL_COUNT_MAX)
    cell_cols = random.randint(WINDOW_CELL_COUNT_MIN, WINDOW_CELL_COUNT_MAX)
    window_h = cell_rows * period + 1
    window_w = cell_cols * period + 1

    hole_h_min = period + 2
    hole_h_max = min(2 * period + 1, window_h // 2 - 1)
    hole_w_min = period + 2
    hole_w_max = min(2 * period + 1, window_w // 2 - 1)
    if hole_h_min > hole_h_max or hole_w_min > hole_w_max:
        return None

    hole_h = random.randint(hole_h_min, hole_h_max)
    hole_w = random.randint(hole_w_min, hole_w_max)
    if hole_h * hole_w <= (period - 1) ** 2:
        return None

    row_choices = _hole_positions(window_h, hole_h, period)
    col_choices = _hole_positions(window_w, hole_w, period)
    if not row_choices or not col_choices:
        return None

    hole_top = random.choice(row_choices)
    hole_left = random.choice(col_choices)

    colors = random.sample(COLOR_POOL, 3 + PATCH_COUNT_MAX)
    bg = colors[0]
    window_color = colors[1]
    sep_color = colors[2]
    extra_colors = colors[3:]

    window, expected_patch = _make_window(
        window_h,
        window_w,
        period,
        hole_top,
        hole_left,
        hole_h,
        hole_w,
        bg,
        window_color,
    )

    masks = _unique_phase_masks(hole_h, hole_w, period)
    masks = [mask for mask in masks if not np.array_equal(mask, expected_patch)]
    patch_count = random.randint(PATCH_COUNT_MIN, PATCH_COUNT_MAX)
    if len(masks) < patch_count - 1:
        return None
    distractors = random.sample(masks, patch_count - 1)
    candidate_masks = [expected_patch.copy()] + [mask.copy() for mask in distractors]
    random.shuffle(candidate_masks)
    if sum(np.array_equal(mask, expected_patch) for mask in candidate_masks) != 1:
        return None
    patch_colors = random.sample(extra_colors, patch_count)
    library = _build_candidate_library(candidate_masks, patch_colors, bg, sep_color)

    top_margin = random.randint(MARGIN_MIN, MARGIN_MAX)
    left_margin = random.randint(MARGIN_MIN, MARGIN_MAX)
    window_top = top_margin
    window_left = left_margin
    library_top = random.randint(MARGIN_MIN, MARGIN_MAX + max(0, window_h - library.shape[0]))
    library_left = window_left + window_w + random.randint(WINDOW_GAP_MIN, WINDOW_GAP_MAX)
    grid_h = max(window_top + window_h, library_top + library.shape[0]) + random.randint(MARGIN_MIN, MARGIN_MAX)
    grid_w = library_left + library.shape[1] + random.randint(MARGIN_MIN, MARGIN_MAX)
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    grid[window_top : window_top + window_h, window_left : window_left + window_w] = window
    grid[library_top : library_top + library.shape[0], library_left : library_left + library.shape[1]] = library

    vals, counts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    expected = _expected_output(
        grid,
        window_top,
        window_left,
        hole_top,
        hole_left,
        expected_patch,
        window_color,
        library_top,
        library_left,
        library.shape,
        bg,
    )
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid window-patch puzzle")
