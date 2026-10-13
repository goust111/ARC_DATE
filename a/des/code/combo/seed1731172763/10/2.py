from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

PANEL_HEIGHT_RANGE = (5, 11)
PANEL_WIDTH_RANGE = (5, 11)
MAX_PANEL_FILL_RATIO = 0.38
PATTERN_OBJECT_RANGE = (1, 3)
RIGHT_OBJECT_RANGE = (2, 4)
DIFF_OBJECT_RANGE = (1, 2)

SHAPE_ATTEMPTS = 40
PLACE_ATTEMPTS = 80
RIGHT_PANEL_ATTEMPTS = 80
GEN_ATTEMPTS = 400

DIFF_MODES = ("paint", "bgify", "recolor", "mixed")


def _mode_color(arr: np.ndarray) -> int:
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert w >= 5 and (w - 2) % 3 == 0

    panel_w = (w - 2) // 3
    sep1 = panel_w
    sep2 = 2 * panel_w + 1
    sep_color = int(g[0, sep1])
    assert sep_color == int(g[0, sep2])
    assert np.all(g[:, sep1] == sep_color)
    assert np.all(g[:, sep2] == sep_color)

    left = g[:, :panel_w]
    middle = g[:, sep1 + 1 : sep2]
    right = g[:, sep2 + 1 :]
    assert left.shape == middle.shape == right.shape == (h, panel_w)

    body = np.concatenate([left, middle, right], axis=1)
    background = _mode_color(body)
    assert background != sep_color
    return left, middle, right, background


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare the first two panels to obtain a difference mask, then crop the third
    panel by keeping only the pixels under that mask.
    """
    left, middle, right, background = _split_panels(grid)
    diff_mask = left != middle
    return np.where(diff_mask, right, background).astype(int)


def _different_non_bg(current: int) -> int:
    choices = [color for color in PALETTE if color != current]
    return int(random.choice(choices))


def _random_shape(max_h: int, max_w: int, max_extent: int, max_cells: int) -> np.ndarray | None:
    kinds: list[str] = []
    if max_h >= 1 and max_w >= 2:
        kinds.append("hline")
    if max_h >= 2 and max_w >= 1:
        kinds.append("vline")
    if max_h >= 2 and max_w >= 2:
        kinds.extend(["rect", "ell"])
    if max_h >= 3 and max_w >= 3:
        kinds.extend(["plus", "ring"])
    if not kinds:
        return None

    for _ in range(SHAPE_ATTEMPTS):
        kind = random.choice(kinds)
        if kind == "hline":
            length = random.randint(2, min(max_w, max_extent + 1))
            mask = np.ones((1, length), dtype=bool)
        elif kind == "vline":
            length = random.randint(2, min(max_h, max_extent + 1))
            mask = np.ones((length, 1), dtype=bool)
        elif kind == "rect":
            h = random.randint(1, min(max_h, max_extent))
            w = random.randint(1, min(max_w, max_extent))
            if h * w < 2:
                continue
            mask = np.ones((h, w), dtype=bool)
        elif kind == "ell":
            h = random.randint(2, min(max_h, max_extent))
            w = random.randint(2, min(max_w, max_extent))
            mask = np.zeros((h, w), dtype=bool)
            mask[:, 0] = True
            mask[-1, :] = True
            mask = np.rot90(mask, random.randrange(4))
        elif kind == "plus":
            size = random.choice([size for size in (3, 5) if size <= min(max_h, max_w, max_extent)])
            mask = np.zeros((size, size), dtype=bool)
            center = size // 2
            mask[center, :] = True
            mask[:, center] = True
        else:
            h = random.randint(3, min(max_h, max_extent))
            w = random.randint(3, min(max_w, max_extent))
            mask = np.ones((h, w), dtype=bool)
            mask[1:-1, 1:-1] = False

        if 2 <= int(mask.sum()) <= max_cells:
            return mask
    return None


def _can_place(blocked: np.ndarray, shape: np.ndarray, top: int, left: int, margin: int) -> bool:
    h, w = shape.shape
    r0 = max(0, top - margin)
    c0 = max(0, left - margin)
    r1 = min(blocked.shape[0], top + h + margin)
    c1 = min(blocked.shape[1], left + w + margin)
    return not bool(blocked[r0:r1, c0:c1].any())


def _stamp(
    panel: np.ndarray,
    blocked: np.ndarray,
    shape: np.ndarray,
    color: int,
    top: int,
    left: int,
    margin: int,
) -> None:
    h, w = shape.shape
    window = panel[top : top + h, left : left + w]
    window[shape] = color

    r0 = max(0, top - margin)
    c0 = max(0, left - margin)
    r1 = min(blocked.shape[0], top + h + margin)
    c1 = min(blocked.shape[1], left + w + margin)
    blocked[r0:r1, c0:c1] = True


def _make_panel(
    h: int,
    w: int,
    object_range: tuple[int, int],
    max_extent: int,
    max_fill_ratio: float,
    margin: int,
) -> np.ndarray | None:
    area = h * w
    panel = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    max_objects = min(object_range[1], max(1, area // 10))
    min_objects = min(object_range[0], max_objects)
    if min_objects > max_objects:
        return None

    target = random.randint(min_objects, max_objects)
    colors = random.sample(PALETTE, random.randint(2, min(4, len(PALETTE))))
    placed = 0

    for _ in range(target):
        success = False
        for _ in range(PLACE_ATTEMPTS):
            shape = _random_shape(h, w, max_extent=max_extent, max_cells=max(4, area // 5))
            if shape is None:
                return None
            sh, sw = shape.shape
            if sh > h or sw > w:
                continue
            if np.count_nonzero(panel) + int(shape.sum()) > int(area * max_fill_ratio):
                continue

            top = random.randint(0, h - sh)
            left = random.randint(0, w - sw)
            if not _can_place(blocked, shape, top, left, margin):
                continue

            _stamp(panel, blocked, shape, random.choice(colors), top, left, margin)
            placed += 1
            success = True
            break
        if not success:
            break

    fill = np.count_nonzero(panel)
    if placed < min_objects or fill < 2 or fill >= area * max_fill_ratio:
        return None
    return panel


def _make_diff_mask(h: int, w: int) -> np.ndarray | None:
    panel = _make_panel(
        h,
        w,
        object_range=DIFF_OBJECT_RANGE,
        max_extent=4,
        max_fill_ratio=0.18,
        margin=1,
    )
    if panel is None:
        return None
    mask = panel != 0
    if int(mask.sum()) < 2 or int(mask.sum()) >= (h * w) // 3:
        return None
    return mask


def _diff_mode_is_valid(left: np.ndarray, diff_mask: np.ndarray, mode: str) -> bool:
    on_fg = int(np.count_nonzero((left != BACKGROUND) & diff_mask))
    on_bg = int(np.count_nonzero((left == BACKGROUND) & diff_mask))
    if mode == "paint":
        return on_bg > 0
    if mode == "bgify":
        return on_fg > 0
    if mode == "recolor":
        return on_fg > 0
    return on_fg > 0 and on_bg > 0


def _apply_difference(left: np.ndarray, diff_mask: np.ndarray, mode: str) -> np.ndarray:
    middle = left.copy()
    for r, c in np.argwhere(diff_mask):
        current = int(middle[r, c])
        if mode == "paint":
            middle[r, c] = _different_non_bg(current)
        elif mode == "bgify":
            middle[r, c] = BACKGROUND if current != BACKGROUND else random.choice(PALETTE)
        elif mode == "recolor":
            middle[r, c] = _different_non_bg(current) if current != BACKGROUND else random.choice(PALETTE)
        else:
            if current == BACKGROUND:
                middle[r, c] = random.choice(PALETTE)
            elif random.random() < 0.5:
                middle[r, c] = BACKGROUND
            else:
                middle[r, c] = _different_non_bg(current)
    return middle


def _assemble(left: np.ndarray, middle: np.ndarray, right: np.ndarray) -> np.ndarray:
    h, w = left.shape
    grid = np.full((h, 3 * w + 2), BACKGROUND, dtype=int)
    grid[:, :w] = left
    grid[:, w] = SEPARATOR
    grid[:, w + 1 : 2 * w + 1] = middle
    grid[:, 2 * w + 1] = SEPARATOR
    grid[:, 2 * w + 2 :] = right
    return grid


def _right_panel_matches_mask(right: np.ndarray, diff_mask: np.ndarray) -> bool:
    right_fg = right != BACKGROUND
    kept = int(np.count_nonzero(right_fg & diff_mask))
    removed = int(np.count_nonzero(right_fg & ~diff_mask))
    return kept > 0 and removed > 0


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*PANEL_HEIGHT_RANGE)
    w = random.randint(*PANEL_WIDTH_RANGE)

    left = _make_panel(
        h,
        w,
        object_range=PATTERN_OBJECT_RANGE,
        max_extent=5,
        max_fill_ratio=MAX_PANEL_FILL_RATIO,
        margin=1,
    )
    if left is None:
        return None

    diff_mask = _make_diff_mask(h, w)
    if diff_mask is None:
        return None

    mode = random.choice(DIFF_MODES)
    if not _diff_mode_is_valid(left, diff_mask, mode):
        return None

    middle = _apply_difference(left, diff_mask, mode)
    if not np.array_equal(left != middle, diff_mask):
        return None

    right = None
    for _ in range(RIGHT_PANEL_ATTEMPTS):
        candidate = _make_panel(
            h,
            w,
            object_range=RIGHT_OBJECT_RANGE,
            max_extent=5,
            max_fill_ratio=MAX_PANEL_FILL_RATIO,
            margin=1,
        )
        if candidate is None:
            continue
        if _right_panel_matches_mask(candidate, diff_mask):
            right = candidate
            break
    if right is None:
        return None

    grid = _assemble(left, middle, right)
    expected = np.where(diff_mask, right, BACKGROUND).astype(int)

    try:
        out = main(grid)
    except Exception:
        return None

    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_ATTEMPTS} attempts")
