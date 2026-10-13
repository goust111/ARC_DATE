from __future__ import annotations

import numpy as np


TOP_COLOR = 1
BOTTOM_COLOR = 2
OVERLAP_COLOR = 4
BACKGROUND = 0

PANEL_SIZE_RANGE = (5, 10)  # sampled as [5, 9]
AREA_RANGE = (5, 11)  # sampled as [5, 10]
MIN_OVERLAP = 2
MAX_OVERLAP = 5
MIN_EXCLUSIVE = 2
MIN_ALT_DIFFERENCE = 4
GEN_ATTEMPTS = 300
MASK_ATTEMPTS = 80
PAIR_ATTEMPTS = 120

_RNG = np.random.default_rng()
_DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the lower panel by 180 degrees, then overlay the two masks with three output colors.

    Input structure:
    - The grid is a 2D array with black background and even height.
    - The top half is one panel containing only black and blue pixels.
    - The bottom half is an equally sized panel containing only black and red pixels.

    Rule:
    - Split the grid across the horizontal midline into the top and bottom panels.
    - Rotate the bottom panel by 180 degrees.
    - Compare the aligned panels cell by cell: blue-only stays blue, red-only stays red, overlap becomes yellow, double-background stays black.

    Key insight:
    - The lower shape is not used as drawn; it must first be rotated as a whole, and the final color depends on whether each aligned cell is occupied by the top mask, the rotated bottom mask, or both.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height % 2 == 0 and width > 0, "expected two equal-height panels"

    half = height // 2
    top = g[:half]
    bottom = g[half:]
    assert top.shape == bottom.shape, "panels must align exactly"
    assert set(map(int, np.unique(top))).issubset({BACKGROUND, TOP_COLOR}), "unexpected top colors"
    assert set(map(int, np.unique(bottom))).issubset({BACKGROUND, BOTTOM_COLOR}), "unexpected bottom colors"

    top_mask = top == TOP_COLOR
    rotated_bottom_mask = np.rot90(bottom == BOTTOM_COLOR, 2)

    out = np.full((half, width), BACKGROUND, dtype=int)
    overlap = top_mask & rotated_bottom_mask
    top_only = top_mask & ~rotated_bottom_mask
    bottom_only = ~top_mask & rotated_bottom_mask
    out[top_only] = TOP_COLOR
    out[bottom_only] = BOTTOM_COLOR
    out[overlap] = OVERLAP_COLOR
    return out


def _neighbors4(r: int, c: int, size: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in _DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            out.append((nr, nc))
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, mask.shape[0]):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _mask_bbox(mask: np.ndarray) -> tuple[int, int]:
    rows, cols = np.nonzero(mask)
    return int(rows.max() - rows.min() + 1), int(cols.max() - cols.min() + 1)


def _is_irregular(mask: np.ndarray) -> bool:
    h, w = _mask_bbox(mask)
    area = int(mask.sum())
    if h < 2 or w < 2:
        return False
    if area == h * w:
        return False
    return True


def _grow_connected_mask(size: int, area: int) -> np.ndarray | None:
    for _ in range(MASK_ATTEMPTS):
        mask = np.zeros((size, size), dtype=bool)
        seed = (int(_RNG.integers(size)), int(_RNG.integers(size)))
        mask[seed] = True
        while int(mask.sum()) < area:
            frontier = set()
            current = np.argwhere(mask)
            for r, c in current:
                for nr, nc in _neighbors4(int(r), int(c), size):
                    if not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break

            frontier_list = list(frontier)
            weights = []
            for cell in frontier_list:
                degree = sum((nnr, nnc) in frontier for nnr, nnc in _neighbors4(cell[0], cell[1], size))
                edge_bias = 1 + int(cell[0] in (0, size - 1)) + int(cell[1] in (0, size - 1))
                weights.append(1.0 + 0.3 * degree + 0.2 * edge_bias)
            pick = frontier_list[int(_RNG.choice(len(frontier_list), p=np.array(weights) / np.sum(weights)))]
            mask[pick] = True

        if int(mask.sum()) == area and _is_connected(mask) and _is_irregular(mask):
            return mask
    return None


def _overlay_from_masks(top_mask: np.ndarray, bottom_mask: np.ndarray) -> np.ndarray:
    out = np.zeros(top_mask.shape, dtype=int)
    overlap = top_mask & bottom_mask
    top_only = top_mask & ~bottom_mask
    bottom_only = ~top_mask & bottom_mask
    out[top_only] = TOP_COLOR
    out[bottom_only] = BOTTOM_COLOR
    out[overlap] = OVERLAP_COLOR
    return out


def _alternative_transforms(bottom_input_mask: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "identity": bottom_input_mask,
        "flipud": np.flipud(bottom_input_mask),
        "fliplr": np.fliplr(bottom_input_mask),
        "rot90": np.rot90(bottom_input_mask, 1),
        "rot180": np.rot90(bottom_input_mask, 2),
        "rot270": np.rot90(bottom_input_mask, 3),
    }


def _pair_is_valid(top_mask: np.ndarray, rotated_bottom_mask: np.ndarray) -> bool:
    if not (_is_connected(top_mask) and _is_connected(rotated_bottom_mask)):
        return False
    if not (_is_irregular(top_mask) and _is_irregular(rotated_bottom_mask)):
        return False
    if np.array_equal(top_mask, rotated_bottom_mask):
        return False

    overlap = int(np.count_nonzero(top_mask & rotated_bottom_mask))
    top_only = int(np.count_nonzero(top_mask & ~rotated_bottom_mask))
    bottom_only = int(np.count_nonzero(~top_mask & rotated_bottom_mask))
    if overlap < MIN_OVERLAP or overlap > MAX_OVERLAP:
        return False
    if top_only < MIN_EXCLUSIVE or bottom_only < MIN_EXCLUSIVE:
        return False

    expected = _overlay_from_masks(top_mask, rotated_bottom_mask)
    if set(map(int, np.unique(expected))) != {BACKGROUND, TOP_COLOR, BOTTOM_COLOR, OVERLAP_COLOR}:
        return False

    bottom_input_mask = np.rot90(rotated_bottom_mask, 2)
    if np.array_equal(bottom_input_mask, rotated_bottom_mask):
        return False

    for name, candidate_mask in _alternative_transforms(bottom_input_mask).items():
        candidate = _overlay_from_masks(top_mask, candidate_mask)
        if name == "rot180":
            if not np.array_equal(candidate, expected):
                return False
            continue
        diff = int(np.count_nonzero(candidate != expected))
        if diff < MIN_ALT_DIFFERENCE:
            return False
    return True


def _build_input(top_mask: np.ndarray, rotated_bottom_mask: np.ndarray) -> np.ndarray:
    size = top_mask.shape[0]
    bottom_input_mask = np.rot90(rotated_bottom_mask, 2)
    inp = np.zeros((size * 2, size), dtype=int)
    top_panel = inp[:size]
    bottom_panel = inp[size:]
    top_panel[top_mask] = TOP_COLOR
    bottom_panel[bottom_input_mask] = BOTTOM_COLOR
    return inp


def _generate_pair(size: int) -> tuple[np.ndarray, np.ndarray] | None:
    max_area = min(AREA_RANGE[1], size * size - 2)
    if max_area < AREA_RANGE[0]:
        return None

    for _ in range(PAIR_ATTEMPTS):
        top_area = int(_RNG.integers(AREA_RANGE[0], max_area + 1))
        bottom_area = int(_RNG.integers(AREA_RANGE[0], max_area + 1))
        top_mask = _grow_connected_mask(size, top_area)
        rotated_bottom_mask = _grow_connected_mask(size, bottom_area)
        if top_mask is None or rotated_bottom_mask is None:
            continue
        if _pair_is_valid(top_mask, rotated_bottom_mask):
            return top_mask, rotated_bottom_mask
    return None


def _semantic_check(inp: np.ndarray) -> bool:
    size = inp.shape[1]
    top_mask = inp[:size] == TOP_COLOR
    bottom_input_mask = inp[size:] == BOTTOM_COLOR
    rotated_bottom_mask = np.rot90(bottom_input_mask, 2)
    if not _pair_is_valid(top_mask, rotated_bottom_mask):
        return False
    try:
        observed = main(inp)
    except Exception:
        return False
    expected = _overlay_from_masks(top_mask, rotated_bottom_mask)
    return bool(np.array_equal(observed, expected))


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        size = int(_RNG.integers(PANEL_SIZE_RANGE[0], PANEL_SIZE_RANGE[1]))
        pair = _generate_pair(size)
        if pair is None:
            continue
        inp = _build_input(*pair)
        if _semantic_check(inp):
            return inp
    raise RuntimeError("failed to generate a valid rotating-panel overlay sample")
