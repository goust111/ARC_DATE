from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists() and str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import bbox, colors_present, get_bg
except Exception:  # pragma: no cover - fallback keeps the file loadable elsewhere
    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])

    def colors_present(grid: np.ndarray, *, exclude=None) -> list[int]:
        cols = set(map(int, np.unique(np.asarray(grid))))
        if exclude is not None:
            cols -= set(map(int, exclude))
        return sorted(cols)

    def bbox(mask: np.ndarray):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)


def _edge_pad(grid: np.ndarray) -> np.ndarray:
    return np.pad(np.asarray(grid, dtype=int), 1, mode="edge")


def _cropped_color_mask(grid: np.ndarray, color: int, bg: int) -> np.ndarray:
    mask = np.asarray(grid, dtype=int) == int(color)
    box = bbox(mask)
    assert box is not None, "expected color to be present"
    r0, c0, r1, c1 = box
    crop = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    crop[mask[r0 : r1 + 1, c0 : c1 + 1]] = int(color)
    return crop


def _is_vertically_symmetric(arr: np.ndarray) -> bool:
    return np.array_equal(arr, np.fliplr(arr))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the only color whose cropped mask becomes vertically symmetric after one-cell edge padding.

    Input structure:
    - The grid has a black background and 3-4 non-background colors.
    - Each color is considered as one full mask, even if it is split into several disconnected pieces.
    - The decisive color touches the left or right border so the edge copy changes its cropped shape.

    Rule:
    - First edge-pad the whole grid by one cell using outward copies of the original border rows, border columns,
      and corner pixels.
    - In the padded grid, crop each non-background color mask to its own bounding box.
    - Test each cropped mask for left-right mirror symmetry and keep the unique symmetric one.
    - Return that cropped mask with the selected color preserved and all other cells black.

    Key insight:
    - The answer is determined only after padding: duplicating boundary pixels can make one border-touching color
      become vertically symmetric even though no color was uniquely symmetric in the original grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    padded = _edge_pad(g)

    symmetric_crops: list[np.ndarray] = []
    for color in colors_present(padded, exclude={bg}):
        crop = _cropped_color_mask(padded, color, bg)
        if _is_vertically_symmetric(crop):
            symmetric_crops.append(crop)

    assert len(symmetric_crops) == 1, "expected exactly one vertically symmetric color mask after padding"
    return symmetric_crops[0]


# --------------- generator ---------------

BG_COLOR = 0
GRID_HEIGHT_RANGE = (8, 11)
GRID_WIDTH_RANGE = (8, 11)
TOTAL_COLORS_CHOICES = (3, 4)
COLOR_POOL = tuple(range(1, 10))
TARGET_KINDS = ("left4", "right4", "left5", "right5")
TARGET_HEIGHT_RANGE = (3, 6)
DISTRACTOR_HEIGHT_RANGE = (2, 5)
DISTRACTOR_WIDTH_RANGE = (3, 5)
INTERIOR_WIDTH_RANGE = (2, 5)
RECT_GAP = 1
MAX_ATTEMPTS = 400
PATCH_ATTEMPTS = 80


def _rect_has_gap(rect: tuple[int, int, int, int], rects: list[tuple[int, int, int, int]], gap: int = RECT_GAP) -> bool:
    r0, c0, r1, c1 = rect
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 + gap < rr0 or rr1 + gap < r0 or c1 + gap < cc0 or cc1 + gap < c0):
            return False
    return True


def _place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    color: int,
    rects: list[tuple[int, int, int, int]],
    *,
    side: str | None = None,
    interior: bool = False,
) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    positions: list[tuple[int, int, tuple[int, int, int, int]]] = []

    if side == "left":
        cols = [0]
    elif side == "right":
        cols = [gw - w]
    elif interior:
        if gw - w < 2:
            return False
        cols = range(1, gw - w)
    else:
        cols = range(0, gw - w + 1)

    for r0 in range(0, gh - h + 1):
        for c0 in cols:
            rect = (r0, c0, r0 + h - 1, c0 + w - 1)
            if _rect_has_gap(rect, rects):
                positions.append((r0, c0, rect))

    if not positions:
        return False

    r0, c0, rect = random.choice(positions)
    view = grid[r0 : r0 + h, c0 : c0 + w]
    view[patch] = int(color)
    rects.append(rect)
    return True


def _build_target_patch(kind: str) -> np.ndarray | None:
    height = random.randint(*TARGET_HEIGHT_RANGE)
    rows: list[tuple[int, int]] = [random.choice(((1, 1), (1, 0), (0, 1))) for _ in range(height)]

    if not any(a == 1 for a, _ in rows):
        idx = random.randrange(height)
        rows[idx] = (1, random.choice((0, 1)))
    if not any(a != b for a, b in rows):
        idx = random.randrange(height)
        rows[idx] = (1, 0)

    patch_rows = []
    for a, b in rows:
        if kind == "left4":
            patch_rows.append([a, b, a, a])
        elif kind == "right4":
            patch_rows.append([a, a, b, a])
        elif kind == "left5":
            patch_rows.append([a, b, b, a, a])
        else:
            patch_rows.append([a, a, b, b, a])

    patch = np.array(patch_rows, dtype=bool)
    if _is_vertically_symmetric(patch.astype(int)):
        return None
    return patch


def _random_asymmetric_patch(height: int, width: int) -> np.ndarray | None:
    for _ in range(PATCH_ATTEMPTS):
        density = random.uniform(0.25, 0.55)
        patch = np.random.random((height, width)) < density
        if int(patch.sum()) < 3 or int(patch.sum()) == height * width:
            continue
        if not patch[0].any() or not patch[-1].any() or not patch[:, 0].any() or not patch[:, -1].any():
            continue
        if _is_vertically_symmetric(patch.astype(int)):
            continue
        return patch
    return None


def _symmetric_color_info(grid: np.ndarray) -> tuple[list[int], list[int]]:
    bg = int(get_bg(grid))
    padded = _edge_pad(grid)
    orig_symmetric = []
    padded_symmetric = []
    for color in colors_present(grid, exclude={bg}):
        original_crop = _cropped_color_mask(grid, color, bg)
        padded_crop = _cropped_color_mask(padded, color, bg)
        if _is_vertically_symmetric(original_crop):
            orig_symmetric.append(color)
        if _is_vertically_symmetric(padded_crop):
            padded_symmetric.append(color)
    return orig_symmetric, padded_symmetric


def _validate_semantics(grid: np.ndarray) -> bool:
    bg = int(get_bg(grid))
    orig_symmetric, padded_symmetric = _symmetric_color_info(grid)
    if orig_symmetric:
        return False
    if len(padded_symmetric) != 1:
        return False

    target = padded_symmetric[0]
    target_coords = np.argwhere(grid == target)
    if not (((target_coords[:, 1] == 0).any()) or ((target_coords[:, 1] == grid.shape[1] - 1).any())):
        return False

    original_crop = _cropped_color_mask(grid, target, bg)
    padded_crop = _cropped_color_mask(_edge_pad(grid), target, bg)
    if original_crop.shape == padded_crop.shape and np.array_equal(original_crop, padded_crop):
        return False

    border_distractors = 0
    for color in colors_present(grid, exclude={bg, target}):
        coords = np.argwhere(grid == color)
        touches_lr_border = ((coords[:, 1] == 0).any()) or ((coords[:, 1] == grid.shape[1] - 1).any())
        if touches_lr_border:
            border_distractors += 1
            original_crop = _cropped_color_mask(grid, color, bg)
            padded_crop = _cropped_color_mask(_edge_pad(grid), color, bg)
            if original_crop.shape == padded_crop.shape and np.array_equal(original_crop, padded_crop):
                return False
            if _is_vertically_symmetric(padded_crop):
                return False

    if border_distractors < 1:
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    expected = _cropped_color_mask(_edge_pad(grid), target, bg)
    return np.array_equal(actual, expected)


def _attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    total_colors = random.choice(TOTAL_COLORS_CHOICES)
    colors = random.sample(COLOR_POOL, total_colors)

    grid = np.full((height, width), BG_COLOR, dtype=int)
    rects: list[tuple[int, int, int, int]] = []

    target_kind = random.choice(TARGET_KINDS)
    target_patch = _build_target_patch(target_kind)
    if target_patch is None:
        return None
    target_side = "left" if "left" in target_kind else "right"
    if not _place_patch(grid, target_patch, colors[0], rects, side=target_side):
        return None

    border_patch_placed = False
    for _ in range(PATCH_ATTEMPTS):
        dh = random.randint(*DISTRACTOR_HEIGHT_RANGE)
        dw = random.randint(DISTRACTOR_WIDTH_RANGE[0], min(DISTRACTOR_WIDTH_RANGE[1], width))
        patch = _random_asymmetric_patch(dh, dw)
        if patch is None:
            continue
        side = random.choice(("left", "right"))
        if _place_patch(grid, patch, colors[1], rects, side=side):
            border_patch_placed = True
            break
    if not border_patch_placed:
        return None

    for color in colors[2:]:
        placed = False
        for _ in range(PATCH_ATTEMPTS):
            dh = random.randint(*DISTRACTOR_HEIGHT_RANGE)
            max_w = min(INTERIOR_WIDTH_RANGE[1], width - 2)
            if max_w < INTERIOR_WIDTH_RANGE[0]:
                return None
            dw = random.randint(INTERIOR_WIDTH_RANGE[0], max_w)
            patch = _random_asymmetric_patch(dh, dw)
            if patch is None:
                continue
            if _place_patch(grid, patch, color, rects, interior=True):
                placed = True
                break
        if not placed:
            return None

    if not _validate_semantics(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid edge-pad symmetry sample")
