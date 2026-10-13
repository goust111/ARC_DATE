from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 300
PLACEMENT_PADDING = 1


def _background_color(grid: np.ndarray) -> int:
    if np.any(grid == BG):
        return BG
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_for_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == color
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _local_shape(coords: Iterable[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    pts = [(int(r), int(c)) for r, c in coords]
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, c0 = min(rows), min(cols)
    local = tuple(sorted((r - r0, c - c0) for r, c in pts))
    return local, max(rows) - r0 + 1, max(cols) - c0 + 1


def _translated_placements(
    shape: tuple[tuple[tuple[int, int], ...], int, int],
    rect_h: int,
    rect_w: int,
) -> list[frozenset[int]]:
    coords, patch_h, patch_w = shape
    placements: list[frozenset[int]] = []
    for r0 in range(rect_h - patch_h + 1):
        for c0 in range(rect_w - patch_w + 1):
            cells = frozenset((r0 + r) * rect_w + (c0 + c) for r, c in coords)
            placements.append(cells)
    return placements


def _rectangle_dims_from_pair(
    shape_a: tuple[tuple[tuple[int, int], ...], int, int],
    shape_b: tuple[tuple[tuple[int, int], ...], int, int],
) -> tuple[int, int]:
    area = len(shape_a[0]) + len(shape_b[0])
    dims: list[tuple[int, int]] = []

    for rect_h in range(1, area + 1):
        if area % rect_h != 0:
            continue
        rect_w = area // rect_h
        if rect_h < max(shape_a[1], shape_b[1]) or rect_w < max(shape_a[2], shape_b[2]):
            continue

        a_places = _translated_placements(shape_a, rect_h, rect_w)
        b_places = _translated_placements(shape_b, rect_h, rect_w)
        found = False
        for a_cells in a_places:
            for b_cells in b_places:
                if a_cells.isdisjoint(b_cells):
                    found = True
                    break
            if found:
                break
        if found:
            dims.append((rect_h, rect_w))

    assert len(dims) == 1, "fragments must reconstruct a rectangle with unique dimensions"
    return dims[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored staircase fragments, rebuild their rectangles, and move each rectangle to its anchor.

    Input structure:
    - The background is black.
    - Each non-background color appears in exactly three 4-connected components: two multi-cell fragments and one isolated single-pixel anchor.
    - The two larger fragments of a color are complementary staircase-cut parts of one filled axis-aligned rectangle, with no rotation needed.

    Rule:
    - For each color, identify its anchor pixel and its two fragments.
    - Find the unique rectangle dimensions for which translations of the two fragments can tile the rectangle with no overlap and no gaps.
    - Erase the original fragments and draw the filled rectangle with its top-left corner at the anchor position; the anchor pixel is covered by the rectangle.

    Key insight:
    - The first step is a shape-completion search: same-colored fragments must be matched by complementary geometry rather than by location.
    - Once the hidden rectangle size is recovered, the singleton anchor supplies the rectangle's new top-left placement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)
    out = np.full_like(g, int(bg))

    colors = [int(color) for color in np.unique(g) if int(color) != int(bg)]
    assert colors, "expected at least one colored fragment pair"

    for color in colors:
        comps = _components_for_color(g, color)
        assert len(comps) == 3, "each color must have two fragments and one anchor"

        anchors = [comp for comp in comps if len(comp) == 1]
        pieces = [comp for comp in comps if len(comp) > 1]
        assert len(anchors) == 1 and len(pieces) == 2, "component sizes do not match puzzle structure"

        anchor_r, anchor_c = anchors[0][0]
        rect_h, rect_w = _rectangle_dims_from_pair(_local_shape(pieces[0]), _local_shape(pieces[1]))

        assert anchor_r + rect_h <= g.shape[0] and anchor_c + rect_w <= g.shape[1], "rectangle exceeds grid"
        region = out[anchor_r : anchor_r + rect_h, anchor_c : anchor_c + rect_w]
        assert np.all(region == bg), "output rectangles must not overlap"
        region[:] = int(color)

    return out


def _random_composition(total: int, parts: int) -> list[int]:
    assert parts >= 1 and total >= parts
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values: list[int] = []
    prev = 0
    for cut in cuts:
        values.append(cut - prev)
        prev = cut
    values.append(total - prev)
    return values


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    assert rows.any() and cols.any()
    r_idx = np.where(rows)[0]
    c_idx = np.where(cols)[0]
    return mask[r_idx[0] : r_idx[-1] + 1, c_idx[0] : c_idx[-1] + 1]


def _sample_split_masks(rect_h: int, rect_w: int) -> tuple[np.ndarray, np.ndarray] | None:
    max_mid = min(rect_h - 2, rect_w - 1, 3)
    if max_mid < 1:
        return None

    mid_count = random.randint(1, max_mid)
    mid_widths = sorted(random.sample(range(1, rect_w), mid_count), reverse=True)
    levels = [rect_w] + mid_widths + [0]
    if rect_h < len(levels):
        return None

    row_runs = _random_composition(rect_h, len(levels))
    widths: list[int] = []
    for width, run in zip(levels, row_runs):
        widths.extend([int(width)] * int(run))

    area_a = sum(widths)
    area_b = rect_h * rect_w - area_a
    if area_a < 3 or area_b < 3:
        return None

    positive_a = [width for width in widths if width > 0]
    positive_b = [rect_w - width for width in widths if width < rect_w]
    if len(set(positive_a)) < 2 or len(set(positive_b)) < 2:
        return None

    mask_a = np.zeros((rect_h, rect_w), dtype=bool)
    for r, width in enumerate(widths):
        if width > 0:
            mask_a[r, :width] = True
    mask_b = ~mask_a
    return mask_a, mask_b


def _mask_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _can_place(reserved: np.ndarray, cells: list[tuple[int, int]], patch_h: int, patch_w: int, r0: int, c0: int) -> bool:
    h, w = reserved.shape
    if r0 < 0 or c0 < 0 or r0 + patch_h > h or c0 + patch_w > w:
        return False
    for dr, dc in cells:
        if reserved[r0 + dr, c0 + dc]:
            return False
    return True


def _candidate_positions(reserved: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    patch_h, patch_w = mask.shape
    cells = _mask_cells(mask)
    candidates: list[tuple[int, int]] = []
    for r0 in range(reserved.shape[0] - patch_h + 1):
        for c0 in range(reserved.shape[1] - patch_w + 1):
            if _can_place(reserved, cells, patch_h, patch_w, r0, c0):
                candidates.append((int(r0), int(c0)))
    random.shuffle(candidates)
    return candidates


def _reserve(reserved: np.ndarray, mask: np.ndarray, r0: int, c0: int, padding: int) -> None:
    h, w = reserved.shape
    for dr, dc in np.argwhere(mask):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        reserved[max(0, rr - padding) : min(h, rr + padding + 1), max(0, cc - padding) : min(w, cc + padding + 1)] = True


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    for dr, dc in np.argwhere(mask):
        grid[int(r0 + dr), int(c0 + dc)] = int(color)


def _place_random(reserved: np.ndarray, mask: np.ndarray, padding: int) -> tuple[int, int] | None:
    candidates = _candidate_positions(reserved, mask)
    if not candidates:
        return None
    r0, c0 = candidates[0]
    _reserve(reserved, mask, r0, c0, padding)
    return r0, c0


def _build_color_spec(color: int, max_dim: int) -> dict[str, object] | None:
    rect_h = random.randint(4, max_dim)
    rect_w = random.randint(4, max_dim)
    split = _sample_split_masks(rect_h, rect_w)
    if split is None:
        return None

    mask_a, mask_b = split
    turns = random.randint(0, 3)
    full_a = np.rot90(mask_a, turns)
    full_b = np.rot90(mask_b, turns)
    piece_a = _trim_mask(full_a)
    piece_b = _trim_mask(full_b)
    out_h, out_w = full_a.shape

    shape_a = _local_shape(_mask_cells(piece_a))
    shape_b = _local_shape(_mask_cells(piece_b))
    try:
        dims = _rectangle_dims_from_pair(shape_a, shape_b)
    except AssertionError:
        return None
    if dims != (out_h, out_w):
        return None

    return {
        "color": int(color),
        "rect_mask": np.ones((out_h, out_w), dtype=bool),
        "piece_masks": [piece_a, piece_b],
    }


def _gen_attempt() -> np.ndarray | None:
    n_colors = random.randint(2, 4)
    min_side = 16 + 3 * n_colors
    grid_h = random.randint(min_side, 30)
    grid_w = random.randint(min_side, 30)
    max_dim = 6 if n_colors == 4 else 7

    palette = random.sample(range(1, 10), n_colors)
    specs: list[dict[str, object]] = []
    for color in palette:
        spec = _build_color_spec(color, max_dim)
        if spec is None:
            return None
        specs.append(spec)

    specs.sort(key=lambda spec: int(np.sum(spec["rect_mask"])), reverse=True)

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    out = np.full((grid_h, grid_w), BG, dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    for spec in specs:
        rect_mask = spec["rect_mask"]
        anchor_pos = _place_random(reserved, rect_mask, PLACEMENT_PADDING)
        if anchor_pos is None:
            return None
        ar, ac = anchor_pos
        color = int(spec["color"])
        spec["anchor"] = (ar, ac)
        inp[ar, ac] = color
        _paint_mask(out, rect_mask, ar, ac, color)

    piece_items: list[tuple[int, np.ndarray]] = []
    for spec in specs:
        color = int(spec["color"])
        for piece_mask in spec["piece_masks"]:
            piece_items.append((color, piece_mask))
    random.shuffle(piece_items)
    piece_items.sort(key=lambda item: int(np.sum(item[1])), reverse=True)

    for color, piece_mask in piece_items:
        pos = _place_random(reserved, piece_mask, PLACEMENT_PADDING)
        if pos is None:
            return None
        pr, pc = pos
        _paint_mask(inp, piece_mask, pr, pc, color)

    if int(np.bincount(inp.ravel()).argmax()) != BG:
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None

    if np.array_equal(inp, out):
        return None
    return inp if np.array_equal(actual, out) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 300 attempts")
