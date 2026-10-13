from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
SEPARATOR_COLORS = (3, 6, 7, 8, 9)

_UNKNOWN_ORDER: list[str] = []
_UNKNOWN_INDEX = 0


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = grid.shape
    assert height >= 5 and width >= 1, "expected three panels and two separator rows"
    assert (height - 2) % 3 == 0, "grid height must be 3*panel_h + 2"

    panel_h = (height - 2) // 3
    sep1 = panel_h
    sep2 = 2 * panel_h + 1

    color1 = int(grid[sep1, 0])
    color2 = int(grid[sep2, 0])
    assert color1 != BLACK and np.all(grid[sep1] == color1), "first separator row must be uniform"
    assert color2 != BLACK and np.all(grid[sep2] == color2), "second separator row must be uniform"

    top = grid[:panel_h, :]
    middle = grid[sep1 + 1 : sep2, :]
    bottom = grid[sep2 + 1 :, :]
    assert top.shape == middle.shape == bottom.shape == (panel_h, width)
    return top, middle, bottom


def _compose_overlay(top_mask: np.ndarray, middle_mask: np.ndarray) -> np.ndarray:
    out = np.full(top_mask.shape, BLACK, dtype=int)
    top_on = top_mask
    middle_on = middle_mask
    out[top_on & ~middle_on] = RED
    out[~top_on & middle_on] = BLUE
    out[top_on & middle_on] = YELLOW
    return out


def _decode_top(bottom_panel: np.ndarray) -> np.ndarray:
    out = np.full(bottom_panel.shape, BLACK, dtype=int)
    out[np.isin(bottom_panel, (RED, YELLOW))] = RED
    return out


def _decode_middle(bottom_panel: np.ndarray) -> np.ndarray:
    out = np.full(bottom_panel.shape, BLACK, dtype=int)
    out[np.isin(bottom_panel, (BLUE, YELLOW))] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the missing panel from a three-panel transparent-overlay equation.

    Input structure:
    - The canvas is split by two full-width separator rows into top, middle, and bottom panels of equal size.
    - The background is black, and exactly one whole panel is replaced by solid grey.
    - Visible top-panel pixels are red, visible middle-panel pixels are blue, and visible bottom-panel pixels use red/blue/yellow overlay encoding.

    Rule:
    - Treat the three panels as one equation: the bottom panel encodes the transparent overlay of the red top mask and blue middle mask.
    - If the grey panel is the bottom one, compose red-only, blue-only, and overlap cells into a red/blue/yellow result.
    - If the grey panel is the top or middle one, invert the encoding from the visible bottom panel: yellow belongs to both operands, so recover red from {red, yellow} and blue from {blue, yellow}.

    Key insight:
    - The equation is fixed, but the solve direction flips with the grey panel position; yellow is not a separate object, it is the shared support that must be returned to both masks during inversion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    top, middle, bottom = _split_panels(g)
    unknown = [
        np.all(top == GREY),
        np.all(middle == GREY),
        np.all(bottom == GREY),
    ]
    assert sum(unknown) == 1, "expected exactly one grey panel"

    if unknown[2]:
        assert set(map(int, np.unique(top))).issubset({BLACK, RED})
        assert set(map(int, np.unique(middle))).issubset({BLACK, BLUE})
        return _compose_overlay(top == RED, middle == BLUE)

    assert set(map(int, np.unique(bottom))).issubset({BLACK, RED, BLUE, YELLOW})
    if unknown[0]:
        return _decode_top(bottom)
    return _decode_middle(bottom)


def _paint_rect(mask: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    mask[r0 : r0 + h, c0 : c0 + w] = True


def _blob_from_center(shape: tuple[int, int], center: tuple[int, int], max_parts: int) -> np.ndarray:
    height, width = shape
    cr, cc = center
    mask = np.zeros(shape, dtype=bool)
    parts = random.randint(1, max_parts)

    for _ in range(parts):
        part_h = random.randint(1, min(4, height))
        part_w = random.randint(1, min(4, width))
        dr = random.randint(-2, 2)
        dc = random.randint(-2, 2)
        r0 = max(0, min(height - part_h, cr + dr - part_h // 2))
        c0 = max(0, min(width - part_w, cc + dc - part_w // 2))
        _paint_rect(mask, r0, c0, part_h, part_w)

    if random.random() < 0.5:
        bar_len = random.randint(2, min(5, max(height, width)))
        if random.random() < 0.5:
            part_h, part_w = 1, min(bar_len, width)
        else:
            part_h, part_w = min(bar_len, height), 1
        r0 = max(0, min(height - part_h, cr - part_h // 2))
        c0 = max(0, min(width - part_w, cc - part_w // 2))
        _paint_rect(mask, r0, c0, part_h, part_w)

    return mask


def _pick_centers(shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    height, width = shape
    pattern = random.choice(("horizontal", "vertical", "diag_down", "diag_up"))

    if pattern == "horizontal":
        return (
            (height // 2, max(1, width // 5)),
            (height // 2, width // 2),
            (height // 2, min(width - 2, (4 * width) // 5)),
        )
    if pattern == "vertical":
        return (
            (max(1, height // 5), width // 2),
            (height // 2, width // 2),
            (min(height - 2, (4 * height) // 5), width // 2),
        )
    if pattern == "diag_down":
        return (
            (max(1, height // 5), max(1, width // 5)),
            (height // 2, width // 2),
            (min(height - 2, (4 * height) // 5), min(width - 2, (4 * width) // 5)),
        )
    return (
        (min(height - 2, (4 * height) // 5), max(1, width // 5)),
        (height // 2, width // 2),
        (max(1, height // 5), min(width - 2, (4 * width) // 5)),
    )


def _generate_masks(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    top_center, overlap_center, middle_center = _pick_centers(shape)

    overlap = _blob_from_center(shape, overlap_center, max_parts=3)
    top_only = _blob_from_center(shape, top_center, max_parts=2)
    middle_only = _blob_from_center(shape, middle_center, max_parts=2)

    top = overlap | top_only
    middle = overlap | middle_only

    if random.random() < 0.4:
        top |= _blob_from_center(shape, top_center, max_parts=1)
    if random.random() < 0.4:
        middle |= _blob_from_center(shape, middle_center, max_parts=1)
    if random.random() < 0.25:
        shared_extra = _blob_from_center(shape, overlap_center, max_parts=1)
        top |= shared_extra
        middle |= shared_extra

    return top, middle


def _mask_semantics_ok(top_mask: np.ndarray, middle_mask: np.ndarray) -> bool:
    top_only = top_mask & ~middle_mask
    middle_only = middle_mask & ~top_mask
    overlap = top_mask & middle_mask
    union = top_mask | middle_mask
    area = int(np.prod(top_mask.shape))

    counts = (
        int(np.count_nonzero(top_only)),
        int(np.count_nonzero(middle_only)),
        int(np.count_nonzero(overlap)),
        int(np.count_nonzero(union)),
    )
    if min(counts[:3]) < 2:
        return False
    if counts[3] < 8 or counts[3] > max(10, area // 2):
        return False
    if int(np.count_nonzero(top_mask)) < 5 or int(np.count_nonzero(middle_mask)) < 5:
        return False
    if np.array_equal(top_mask, middle_mask):
        return False

    rows, cols = np.where(union)
    if rows.size == 0:
        return False
    if rows.min() == 0 and rows.max() == top_mask.shape[0] - 1 and cols.min() == 0 and cols.max() == top_mask.shape[1] - 1:
        return False
    return True


def _render_mask(mask: np.ndarray, color: int) -> np.ndarray:
    panel = np.full(mask.shape, BLACK, dtype=int)
    panel[mask] = color
    return panel


def _assemble_input(top_panel: np.ndarray, middle_panel: np.ndarray, bottom_panel: np.ndarray, unknown: str) -> np.ndarray:
    panel_h, panel_w = top_panel.shape
    sep1 = random.choice(SEPARATOR_COLORS)
    sep2 = random.choice(SEPARATOR_COLORS)
    unknown_panel = np.full((panel_h, panel_w), GREY, dtype=int)

    if unknown == "top":
        top_panel = unknown_panel
    elif unknown == "middle":
        middle_panel = unknown_panel
    else:
        bottom_panel = unknown_panel

    grid = np.full((panel_h * 3 + 2, panel_w), BLACK, dtype=int)
    grid[:panel_h, :] = top_panel
    grid[panel_h, :] = sep1
    grid[panel_h + 1 : 2 * panel_h + 1, :] = middle_panel
    grid[2 * panel_h + 1, :] = sep2
    grid[2 * panel_h + 2 :, :] = bottom_panel
    return grid


def _semantic_sample_ok(grid: np.ndarray, expected: np.ndarray, unknown: str) -> bool:
    try:
        top, middle, bottom = _split_panels(grid)
    except AssertionError:
        return False

    unknown_flags = [np.all(top == GREY), np.all(middle == GREY), np.all(bottom == GREY)]
    if sum(unknown_flags) != 1:
        return False

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    if unknown == "top":
        if not np.any(bottom == RED) or not np.any(bottom == BLUE) or not np.any(bottom == YELLOW):
            return False
        if not set(map(int, np.unique(expected))).issubset({BLACK, RED}):
            return False
    elif unknown == "middle":
        if not np.any(bottom == RED) or not np.any(bottom == BLUE) or not np.any(bottom == YELLOW):
            return False
        if not set(map(int, np.unique(expected))).issubset({BLACK, BLUE}):
            return False
    else:
        if not set(map(int, np.unique(expected))).issubset({BLACK, RED, BLUE, YELLOW}):
            return False
        if not np.any(expected == RED) or not np.any(expected == BLUE) or not np.any(expected == YELLOW):
            return False

    return True


def _next_unknown() -> str:
    global _UNKNOWN_INDEX, _UNKNOWN_ORDER
    if _UNKNOWN_INDEX % 3 == 0:
        _UNKNOWN_ORDER = ["top", "middle", "bottom"]
        random.shuffle(_UNKNOWN_ORDER)
    unknown = _UNKNOWN_ORDER[_UNKNOWN_INDEX % 3]
    _UNKNOWN_INDEX += 1
    return unknown


def _gen_attempt(unknown: str) -> np.ndarray | None:
    panel_h = random.randint(5, 9)
    panel_w = random.randint(6, 13)
    shape = (panel_h, panel_w)

    top_mask, middle_mask = _generate_masks(shape)
    if not _mask_semantics_ok(top_mask, middle_mask):
        return None

    top_panel = _render_mask(top_mask, RED)
    middle_panel = _render_mask(middle_mask, BLUE)
    bottom_panel = _compose_overlay(top_mask, middle_mask)

    if unknown == "top":
        expected = top_panel
    elif unknown == "middle":
        expected = middle_panel
    else:
        expected = bottom_panel

    grid = _assemble_input(top_panel, middle_panel, bottom_panel, unknown)
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if not _semantic_sample_ok(grid, expected, unknown):
        return None
    return grid


def generate_input() -> np.ndarray:
    unknown = _next_unknown()
    for _ in range(400):
        grid = _gen_attempt(unknown)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid three-color-equation sample")
