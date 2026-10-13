from __future__ import annotations

import hashlib
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from arc_utils import bbox, components_mask, crop_bbox, get_bg, paste


BG = 0
TEMPLATE_COLOR = 5
MARKER_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
TEMPLATE_SHAPES = ((4, 4), (4, 5), (5, 4), (5, 5))

MAX_GEN_ATTEMPTS = 600
_SEEN_OUTPUTS: set[str] = set()
_BRANCH_COUNTS = {False: 0, True: 0}


def _hole_components(mask: np.ndarray) -> list[np.ndarray]:
    holes: list[np.ndarray] = []
    for comp in components_mask(~mask, connectivity=4):
        touches_edge = np.any(
            (comp[:, 0] == 0)
            | (comp[:, 1] == 0)
            | (comp[:, 0] == mask.shape[0] - 1)
            | (comp[:, 1] == mask.shape[1] - 1)
        )
        if not bool(touches_edge):
            holes.append(comp)
    return holes


def _edge_background_count(mask: np.ndarray) -> int:
    edge = np.zeros_like(mask, dtype=bool)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True
    return int(np.count_nonzero((~mask) & edge))


def _sorted_markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    counts = Counter(map(int, grid.ravel()))
    markers: list[tuple[int, int, int]] = []
    for color, count in counts.items():
        if color in (BG, TEMPLATE_COLOR):
            continue
        assert count == 1, "every non-template foreground color must be a single marker"
        r, c = map(int, np.argwhere(grid == color)[0])
        markers.append((r, c, int(color)))

    assert 3 <= len(markers) <= 5, "expected 3-5 markers"
    assert len({r for r, _c, _color in markers}) == 1, "markers must share one row"
    markers.sort(key=lambda item: item[1])
    return markers


def _template_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    template_bbox = bbox(g == TEMPLATE_COLOR)
    assert template_bbox is not None, "missing gray template"
    mask = crop_bbox(g == TEMPLATE_COLOR, template_bbox)
    h, w = map(int, mask.shape)
    assert (h, w) in TEMPLATE_SHAPES, f"unexpected template bbox: {(h, w)}"
    assert len(components_mask(mask, connectivity=4)) == 1, "template must be 4-connected"
    return mask


def _render_output(mask: np.ndarray, colors: list[int], invert: bool) -> np.ndarray:
    h, w = map(int, mask.shape)
    paint_mask = ~mask if invert else mask
    out = np.full((h, w * len(colors)), BG, dtype=int)
    for idx, color in enumerate(colors):
        patch = np.full((h, w), BG, dtype=int)
        patch[paint_mask] = int(color)
        paste(out, patch, 0, idx * w)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Concatenate template copies whose coloring is globally flipped when the template encloses a hole.

    Input structure:
    - A black background contains one gray 4-connected template in the upper region.
    - The template's tight 4x4 to 5x5 bounding box may contain background concavities, and may also contain exactly one enclosed 4-connected black hole.
    - A lower row contains 3-5 distinct single-pixel markers of different colors.

    Rule:
    - Crop the gray template to its tight bounding box and use gray cells as the template mask.
    - Detect whether the cropped mask contains an enclosed 4-connected background hole.
    - Read the marker colors from left to right.
    - If there is no hole, paint each copy on the template foreground cells.
    - If there is a hole, invert every copy: paint the template background cells inside the bbox, including the hole, and leave the gray-mask cells black.
    - Concatenate all copies horizontally with no gaps.

    Key insight:
    - Marker order chooses copy order, but only the template's topology decides whether every copy is rendered normally or with foreground/background swapped.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(get_bg(g)) == BG, "background must be black"

    mask = _template_mask(g)
    markers = _sorted_markers(g)
    colors = [color for _r, _c, color in markers]
    invert = bool(_hole_components(mask))
    return _render_output(mask, colors, invert)


def _connected(mask: np.ndarray) -> bool:
    return len(components_mask(mask, connectivity=4)) == 1


def _fills_bbox(mask: np.ndarray) -> bool:
    h, w = mask.shape
    return bool(
        np.any(mask[0, :])
        and np.any(mask[-1, :])
        and np.any(mask[:, 0])
        and np.any(mask[:, -1])
        and bbox(mask) == (0, 0, h - 1, w - 1)
    )


def _carve_border_notch(mask: np.ndarray) -> None:
    h, w = map(int, mask.shape)
    side = random.choice(("top", "bottom", "left", "right"))
    if side in ("top", "bottom"):
        depth = random.randint(1, h - 2)
        span = random.randint(1, w - 1)
        c0 = random.randint(0, w - span)
        if side == "top":
            mask[:depth, c0 : c0 + span] = False
        else:
            mask[h - depth :, c0 : c0 + span] = False
    else:
        depth = random.randint(1, w - 2)
        span = random.randint(1, h - 1)
        r0 = random.randint(0, h - span)
        if side == "left":
            mask[r0 : r0 + span, :depth] = False
        else:
            mask[r0 : r0 + span, w - depth :] = False


def _validate_mask(mask: np.ndarray, want_hole: bool) -> bool:
    area = int(np.count_nonzero(mask))
    background_area = int(np.count_nonzero(~mask))
    if not (mask.any() and (~mask).any()):
        return False
    if area < 8 or background_area < 2:
        return False
    if not _fills_bbox(mask):
        return False
    if not _connected(mask):
        return False
    if _edge_background_count(mask) < 1:
        return False

    holes = _hole_components(mask)
    if want_hole:
        if len(holes) != 1:
            return False
        if background_area < 3:
            return False
    elif holes:
        return False

    return True


def _generate_mask(shape: tuple[int, int], want_hole: bool) -> np.ndarray:
    h, w = map(int, shape)
    for _ in range(400):
        mask = np.ones((h, w), dtype=bool)

        if want_hole:
            hole_h = random.randint(1, h - 2)
            hole_w = random.randint(1, w - 2)
            hole_r = random.randint(1, h - hole_h - 1)
            hole_c = random.randint(1, w - hole_w - 1)
            mask[hole_r : hole_r + hole_h, hole_c : hole_c + hole_w] = False
            notch_count = random.randint(1, 2)
        else:
            notch_count = random.randint(1, 3)

        for _ in range(notch_count):
            _carve_border_notch(mask)

        if not _validate_mask(mask, want_hole):
            continue
        return mask

    raise RuntimeError("failed to generate a valid template mask")


def _choose_branch() -> bool:
    if _BRANCH_COUNTS[True] < _BRANCH_COUNTS[False]:
        return True
    if _BRANCH_COUNTS[False] < _BRANCH_COUNTS[True]:
        return False
    return bool(random.getrandbits(1))


def _marker_sequence(count: int) -> list[int]:
    for _ in range(100):
        colors = random.sample(MARKER_PALETTE, count)
        if colors != sorted(colors) and colors != sorted(colors, reverse=True):
            return colors
    raise RuntimeError("failed to sample a non-monotone marker sequence")


def _marker_columns(width: int, count: int) -> list[int]:
    usable = list(range(1, width - 1))
    for _ in range(200):
        cols = sorted(random.sample(usable, count))
        gaps = [cols[i + 1] - cols[i] for i in range(count - 1)]
        if min(gaps) < 2:
            continue
        if len(set(gaps)) == 1:
            continue
        return cols
    raise RuntimeError("failed to sample irregular marker columns")


def _build_input(mask: np.ndarray, marker_colors: list[int]) -> np.ndarray:
    h, w = map(int, mask.shape)
    count = len(marker_colors)

    min_h = h + 7
    min_w = max(14, w + count + 7)
    grid_h = random.randint(min_h, 18)
    grid_w = random.randint(min_w, 20)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    template_r = random.randint(1, 3)
    template_c = random.randint(1, grid_w - w - 1)
    grid[template_r : template_r + h, template_c : template_c + w][mask] = TEMPLATE_COLOR

    marker_row = random.randint(template_r + h + 2, grid_h - 2)
    for col, color in zip(_marker_columns(grid_w, count), marker_colors):
        grid[marker_row, col] = int(color)

    return grid


def _signature(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes() + str(arr.shape).encode()).hexdigest()


def _validate_example(grid: np.ndarray, want_hole: bool) -> bool:
    try:
        mask = _template_mask(grid)
        markers = _sorted_markers(grid)
    except AssertionError:
        return False

    holes = _hole_components(mask)
    if want_hole != bool(holes):
        return False
    if want_hole and len(holes) != 1:
        return False
    if _edge_background_count(mask) < 1:
        return False
    if int(np.count_nonzero(~mask)) < 2:
        return False

    colors = [color for _r, _c, color in markers]
    actual = main(grid)
    expected = _render_output(mask, colors, want_hole)
    if not np.array_equal(actual, expected):
        return False

    tile = actual[:, : mask.shape[1]]
    if want_hole:
        if not np.array_equal(tile != BG, ~mask):
            return False
    else:
        if not np.array_equal(tile != BG, mask):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        want_hole = _choose_branch()
        shape = random.choice(TEMPLATE_SHAPES)
        marker_count = random.randint(3, 5)
        mask = _generate_mask(shape, want_hole)
        marker_colors = _marker_sequence(marker_count)
        grid = _build_input(mask, marker_colors)
        if not _validate_example(grid, want_hole):
            continue

        out_sig = _signature(main(grid))
        if out_sig in _SEEN_OUTPUTS:
            continue

        _SEEN_OUTPUTS.add(out_sig)
        _BRANCH_COUNTS[want_hole] += 1
        return grid

    raise RuntimeError("failed to generate a valid input for puzzle 7")
