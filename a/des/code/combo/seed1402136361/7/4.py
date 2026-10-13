from __future__ import annotations

import numpy as np


def _unique_dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    max_count = int(counts.max())
    winners = values[counts == max_count]
    assert len(winners) == 1, "background must be uniquely dominant"
    return int(winners[0])


def _find_separator_column(grid: np.ndarray, bg: int) -> int:
    height, width = grid.shape
    assert width % 2 == 1, "two equal panels plus one separator require odd width"

    candidates = []
    for c in range(width):
        if c != width - c - 1:
            continue
        if np.all(grid[:, c] == grid[0, c]) and int(grid[0, c]) != bg:
            candidates.append(c)

    assert len(candidates) == 1, "expected exactly one centered separator column"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Zipper-merge the two side panels row by row.

    Input structure
    - The background color is dominant.
    - One full-height separator column splits the grid into equal left/right panels.
    - Both panels may contain arbitrary foreground patterns on the shared background.

    Rule
    - Remove the separator column.
    - Output odd-numbered rows (1-based) from the left panel.
    - Output even-numbered rows from the right panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _unique_dominant_color(g)
    sep_col = _find_separator_column(g, bg)
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]

    assert left.shape == right.shape and left.size > 0, "panels must be equal and non-empty"
    assert np.any(left != bg) and np.any(right != bg), "both panels must contain foreground"

    out = right.copy()
    out[0::2] = left[0::2]
    out[1::2] = right[1::2]
    return out


# --------------- generator ---------------

COLOR_IDS = tuple(range(10))
MIN_HEIGHT = 6
MAX_HEIGHT = 12
MIN_PANEL_WIDTH = 4
MAX_PANEL_WIDTH = 10
MIN_ACTIVE_COLORS = 3
MAX_ACTIVE_COLORS = 6
MIN_OBJECTS_PER_PANEL = 2
MAX_OBJECTS_PER_PANEL = 6
MAX_FILL_RATIO = 0.35
PADDING = 1
OBJECT_PLACEMENT_TRIES = 80
PANEL_BUILD_TRIES = 120
MAX_GENERATION_ATTEMPTS = 300


def _shape_pixel() -> np.ndarray:
    return np.array([[True]], dtype=bool)


def _shape_hline(length: int) -> np.ndarray:
    return np.ones((1, length), dtype=bool)


def _shape_vline(length: int) -> np.ndarray:
    return np.ones((length, 1), dtype=bool)


def _shape_rect(height: int, width: int) -> np.ndarray:
    return np.ones((height, width), dtype=bool)


def _shape_elbow(height: int, width: int, orientation: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if orientation == 0:
        mask[0, :] = True
        mask[:, 0] = True
    elif orientation == 1:
        mask[0, :] = True
        mask[:, -1] = True
    elif orientation == 2:
        mask[-1, :] = True
        mask[:, 0] = True
    else:
        mask[-1, :] = True
        mask[:, -1] = True
    return mask


def _shape_plus() -> np.ndarray:
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, :] = True
    mask[:, 1] = True
    return mask


def _shape_tee(orientation: int) -> np.ndarray:
    mask = np.zeros((3, 3), dtype=bool)
    if orientation == 0:
        mask[0, :] = True
        mask[:, 1] = True
    elif orientation == 1:
        mask[:, -1] = True
        mask[1, :] = True
    elif orientation == 2:
        mask[-1, :] = True
        mask[:, 1] = True
    else:
        mask[:, 0] = True
        mask[1, :] = True
    return mask


def _sample_shape(rng: np.random.Generator, panel_h: int, panel_w: int) -> np.ndarray:
    builders: list[str] = ["pixel"]
    if panel_w >= 2:
        builders.append("hline")
    if panel_h >= 2:
        builders.append("vline")
    if panel_h >= 2 and panel_w >= 2:
        builders.extend(["rect", "elbow"])
    if panel_h >= 3 and panel_w >= 3:
        builders.extend(["plus", "tee"])

    kind = str(rng.choice(builders))
    if kind == "pixel":
        return _shape_pixel()
    if kind == "hline":
        return _shape_hline(int(rng.integers(2, min(4, panel_w) + 1)))
    if kind == "vline":
        return _shape_vline(int(rng.integers(2, min(4, panel_h) + 1)))
    if kind == "rect":
        rect_h = int(rng.integers(2, min(3, panel_h) + 1))
        rect_w = int(rng.integers(2, min(4, panel_w) + 1))
        return _shape_rect(rect_h, rect_w)
    if kind == "elbow":
        elbow_h = int(rng.integers(2, min(4, panel_h) + 1))
        elbow_w = int(rng.integers(2, min(4, panel_w) + 1))
        return _shape_elbow(elbow_h, elbow_w, int(rng.integers(0, 4)))
    if kind == "plus":
        return _shape_plus()
    return _shape_tee(int(rng.integers(0, 4)))


def _can_place(mask: np.ndarray, occupied: np.ndarray, top: int, left: int, padding: int = PADDING) -> bool:
    height, width = occupied.shape
    mask_h, mask_w = mask.shape
    if top + mask_h > height or left + mask_w > width:
        return False

    r0 = max(0, top - padding)
    r1 = min(height, top + mask_h + padding)
    c0 = max(0, left - padding)
    c1 = min(width, left + mask_w + padding)
    return not np.any(occupied[r0:r1, c0:c1])


def _place_mask(panel: np.ndarray, occupied: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    mask_h, mask_w = mask.shape
    window = panel[top : top + mask_h, left : left + mask_w]
    window[mask] = color
    occupied[top : top + mask_h, left : left + mask_w] |= mask


def _build_panel(
    rng: np.random.Generator,
    panel_h: int,
    panel_w: int,
    bg: int,
    palette: list[int],
) -> np.ndarray | None:
    panel = np.full((panel_h, panel_w), bg, dtype=int)
    occupied = np.zeros((panel_h, panel_w), dtype=bool)

    area = panel_h * panel_w
    max_objects = min(MAX_OBJECTS_PER_PANEL, max(MIN_OBJECTS_PER_PANEL, area // 10))
    target_objects = int(rng.integers(MIN_OBJECTS_PER_PANEL, max_objects + 1))
    placed = 0

    for _ in range(PANEL_BUILD_TRIES):
        if placed >= target_objects:
            break
        mask = _sample_shape(rng, panel_h, panel_w)
        mask_h, mask_w = mask.shape
        if mask_h > panel_h or mask_w > panel_w:
            continue

        color = int(rng.choice(palette))
        for _ in range(OBJECT_PLACEMENT_TRIES):
            top = int(rng.integers(0, panel_h - mask_h + 1))
            left = int(rng.integers(0, panel_w - mask_w + 1))
            if not _can_place(mask, occupied, top, left):
                continue
            _place_mask(panel, occupied, mask, top, left, color)
            placed += 1
            break

    filled = int(np.count_nonzero(panel != bg))
    if placed < MIN_OBJECTS_PER_PANEL or filled == 0:
        return None
    if filled / area > MAX_FILL_RATIO:
        return None
    return panel


def _expected_output(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = right.copy()
    out[0::2] = left[0::2]
    out[1::2] = right[1::2]
    return out


def _rows_differ_on_parity(left: np.ndarray, right: np.ndarray, parity: int) -> bool:
    return any(not np.array_equal(left[r], right[r]) for r in range(parity, left.shape[0], 2))


def _has_foreground_on_parity(panel: np.ndarray, bg: int, parity: int) -> bool:
    return any(np.any(panel[r] != bg) for r in range(parity, panel.shape[0], 2))


def _is_valid_instance(grid: np.ndarray, left: np.ndarray, right: np.ndarray, bg: int) -> bool:
    if not _has_foreground_on_parity(left, bg, 0):
        return False
    if not _has_foreground_on_parity(right, bg, 1):
        return False
    if not _rows_differ_on_parity(left, right, 0):
        return False
    if not _rows_differ_on_parity(left, right, 1):
        return False

    values, counts = np.unique(grid, return_counts=True)
    max_count = int(counts.max())
    dominant = values[counts == max_count]
    if len(dominant) != 1 or int(dominant[0]) != bg:
        return False

    expected = _expected_output(left, right)
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    panel_h = int(rng.integers(MIN_HEIGHT, MAX_HEIGHT + 1))
    panel_w = int(rng.integers(MIN_PANEL_WIDTH, MAX_PANEL_WIDTH + 1))
    bg, sep = map(int, rng.choice(COLOR_IDS, size=2, replace=False))
    palette_pool = [color for color in COLOR_IDS if color not in {bg, sep}]
    palette_size = int(rng.integers(MIN_ACTIVE_COLORS, min(MAX_ACTIVE_COLORS, len(palette_pool)) + 1))
    palette = list(map(int, rng.choice(palette_pool, size=palette_size, replace=False)))

    left = _build_panel(rng, panel_h, panel_w, bg, palette)
    right = _build_panel(rng, panel_h, panel_w, bg, palette)
    if left is None or right is None:
        return None

    grid = np.full((panel_h, 2 * panel_w + 1), bg, dtype=int)
    grid[:, :panel_w] = left
    grid[:, panel_w] = sep
    grid[:, panel_w + 1 :] = right

    if not _is_valid_instance(grid, left, right, bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid zipper-panels input")
