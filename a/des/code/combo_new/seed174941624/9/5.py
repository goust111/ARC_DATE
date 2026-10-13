import itertools
import random

import numpy as np


Rect = tuple[int, int, int, int]

_BG = 0
_PALETTE = tuple(range(1, 10))
_LAYER_OPTIONS = (2, 3, 4)
_BLOCK_H_MAX = {2: 10, 3: 8, 4: 7}
_BLOCK_H_MIN = 4
_BLOCK_W_RANGE = (5, 12)
_MARGIN_RANGE = (1, 4)
_GEN_RETRIES = 400
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bbox_from_mask(mask: np.ndarray) -> Rect:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected at least one marked cell"
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _rect_area(rect: Rect) -> int:
    r0, c0, r1, c1 = rect
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _render_rectangles(
    shape: tuple[int, int], rect_by_color: dict[int, Rect], order: tuple[int, ...] | list[int]
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for color in order:
        r0, c0, r1, c1 = rect_by_color[int(color)]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def _stack_layers(shape: tuple[int, int], rect_by_color: dict[int, Rect], order: tuple[int, ...]) -> np.ndarray:
    h, w = map(int, shape)
    out = np.zeros((h * len(order), w), dtype=int)
    for layer_idx, color in enumerate(order):
        r0, c0, r1, c1 = rect_by_color[int(color)]
        base = layer_idx * h
        out[base + r0 : base + r1 + 1, c0 : c1 + 1] = int(color)
    return out


def _crop_nonzero(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    coords = np.argwhere(grid != _BG)
    assert coords.size > 0, "expected a non-empty foreground block"
    r0, c0, r1, c1 = _bbox_from_mask(grid != _BG)
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], copy=True), (int(r0), int(c0))


def _matching_orders(patch: np.ndarray) -> tuple[dict[int, Rect], list[tuple[int, ...]]]:
    colors = [int(c) for c in np.unique(patch) if int(c) != _BG]
    rect_by_color = {color: _bbox_from_mask(patch == color) for color in colors}
    matches: list[tuple[int, ...]] = []
    for order in itertools.permutations(colors):
        rendered = _render_rectangles(patch.shape, rect_by_color, order)
        if np.array_equal(rendered, patch):
            matches.append(tuple(int(c) for c in order))
    return rect_by_color, matches


def _count_components(mask: np.ndarray) -> int:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                for dr, dc in _NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _edge_visibility_preserved(parent: Rect, child: Rect) -> bool:
    pr0, pc0, pr1, pc1 = parent
    cr0, cc0, cr1, cc1 = child
    top_hidden = cr0 == pr0 and cc0 == pc0 and cc1 == pc1
    bottom_hidden = cr1 == pr1 and cc0 == pc0 and cc1 == pc1
    left_hidden = cc0 == pc0 and cr0 == pr0 and cr1 == pr1
    right_hidden = cc1 == pc1 and cr0 == pr0 and cr1 == pr1
    return not (top_hidden or bottom_hidden or left_hidden or right_hidden)


def _child_candidates(parent: Rect, min_h: int, min_w: int) -> list[Rect]:
    pr0, pc0, pr1, pc1 = parent
    ph = int(pr1 - pr0 + 1)
    pw = int(pc1 - pc0 + 1)
    candidates: list[Rect] = []
    for h in range(int(min_h), ph + 1):
        for w in range(int(min_w), pw + 1):
            for dr in range(ph - h + 1):
                for dc in range(pw - w + 1):
                    child = (pr0 + dr, pc0 + dc, pr0 + dr + h - 1, pc0 + dc + w - 1)
                    if child == parent:
                        continue
                    if not _edge_visibility_preserved(parent, child):
                        continue
                    candidates.append(child)
    return candidates


def _build_nested_chain(parent: Rect, layers_above: int) -> list[Rect] | None:
    if layers_above == 0:
        return []

    min_h = 2 if layers_above == 1 else 3
    min_w = 2 if layers_above == 1 else 3
    candidates = _child_candidates(parent, min_h, min_w)
    random.shuffle(candidates)

    for child in candidates:
        suffix = _build_nested_chain(child, layers_above - 1)
        if suffix is not None:
            return [child] + suffix
    return None


def _has_occlusion_witness(mask: np.ndarray, rect: Rect) -> bool:
    r0, c0, r1, c1 = rect
    full = np.zeros_like(mask, dtype=bool)
    full[r0 : r1 + 1, c0 : c1 + 1] = True
    missing = full & ~mask
    if not np.any(missing):
        return False
    for r, c in np.argwhere(missing):
        for dr, dc in _NEIGHBORS4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                return True
    return False


def _instance_is_valid(patch: np.ndarray, rect_by_color: dict[int, Rect], order: tuple[int, ...]) -> bool:
    if _count_components(patch != _BG) != 1:
        return False
    if patch.shape[0] * len(order) > 30 or patch.shape[1] > 30:
        return False

    inferred_rects, matches = _matching_orders(patch)
    if matches != [order]:
        return False
    if inferred_rects != rect_by_color:
        return False

    for idx, color in enumerate(order):
        mask = patch == int(color)
        if int(mask.sum()) < 4:
            return False
        if _bbox_from_mask(mask) != rect_by_color[int(color)]:
            return False
        if idx == len(order) - 1:
            if int(mask.sum()) != _rect_area(rect_by_color[int(color)]):
                return False
        else:
            if int(mask.sum()) >= _rect_area(rect_by_color[int(color)]):
                return False
            if not _has_occlusion_witness(mask, rect_by_color[int(color)]):
                return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Separate one overlapped multi-color rectangle block into bottom-to-top monochrome panels.

    Input structure:
    - Black background surrounds one compact multi-color block.
    - Each non-black color is the visible remnant of one solid axis-aligned rectangle.
    - Later rectangles partially cover earlier ones, so lower layers can look jagged or bitten.

    Rule:
    - Crop to the bounding box of the non-black block.
    - For each color, take the bounding box of its visible pixels as that layer's original full rectangle.
    - Try every possible bottom-to-top drawing order and keep the unique order whose full-rectangle redraw exactly reproduces the cropped input.
    - Output one black-backed panel per layer, stacked from top to bottom in bottom-layer to top-layer order.

    Key insight:
    - Occlusion removes interior chunks, but the extreme rows and columns of each color still reveal its hidden full rectangle; the correct layer order is the only redraw that recreates the observed overlaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    patch, _offset = _crop_nonzero(g)
    colors = [int(c) for c in np.unique(patch) if int(c) != _BG]
    assert 2 <= len(colors) <= 4, "expected a 2-4 color block"

    rect_by_color, matches = _matching_orders(patch)
    assert len(matches) == 1, "layer order must be uniquely determined"
    order = matches[0]
    return _stack_layers(patch.shape, rect_by_color, order)


def _gen_attempt() -> np.ndarray | None:
    n_layers = random.choice(_LAYER_OPTIONS)
    block_h = random.randint(_BLOCK_H_MIN, _BLOCK_H_MAX[n_layers])
    block_w = random.randint(*_BLOCK_W_RANGE)
    colors = tuple(int(c) for c in random.sample(_PALETTE, n_layers))

    bottom = (0, 0, block_h - 1, block_w - 1)
    chain = _build_nested_chain(bottom, n_layers - 1)
    if chain is None:
        return None

    rects = [bottom] + chain
    rect_by_color = {color: rect for color, rect in zip(colors, rects)}
    patch = _render_rectangles((block_h, block_w), rect_by_color, colors)

    if not _instance_is_valid(patch, rect_by_color, colors):
        return None

    expected = _stack_layers(patch.shape, rect_by_color, colors)

    top_margin = random.randint(*_MARGIN_RANGE)
    bottom_margin = random.randint(*_MARGIN_RANGE)
    left_margin = random.randint(*_MARGIN_RANGE)
    right_margin = random.randint(*_MARGIN_RANGE)

    grid_h = block_h + top_margin + bottom_margin
    grid_w = block_w + left_margin + right_margin
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[top_margin : top_margin + block_h, left_margin : left_margin + block_w] = patch

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid occlusion-layer puzzle")
