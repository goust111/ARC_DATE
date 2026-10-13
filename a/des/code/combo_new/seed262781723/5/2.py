from __future__ import annotations

from collections import deque

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble each same-colored fragment pair into a solid rectangle, then pack the rectangles by increasing area.

    Input structure:
    - The background is black.
    - Each non-black color appears in exactly two separate 4-connected fragments.
    - The two fragments of a color are translation-only pieces of one solid rectangle; no rotation or reflection is needed.

    Rule:
    - For each color, find its two fragments and determine the unique rectangle height and width that the two shapes can fill together without overlap after translating one fragment.
    - Sort the reconstructed rectangles by area from smallest to largest.
    - Create an output grid whose height is the maximum rectangle height and whose width is the sum of rectangle widths plus one black separator column between consecutive rectangles.
    - Paint each rectangle in its original color, top-aligned, from left to right in the sorted order.

    Key insight:
    - The visible shapes are incomplete complementary pieces: the task is to infer the hidden full rectangle for each color by testing translation-only assembly, then ignore the original locations and reorder the completed rectangles by area.
    """
    g = np.asarray(grid, dtype=int)
    colors = sorted(int(v) for v in np.unique(g) if int(v) != 0)
    assert colors

    rectangles: list[tuple[int, int, int, int]] = []
    for color in colors:
        comps = _components_of_color(g, color)
        assert len(comps) == 2, f"color {color} must have exactly two fragments"
        dims = _pair_rectangle_dims(comps[0], comps[1])
        assert dims is not None, f"color {color} does not form a unique rectangle"
        h, w = dims
        rectangles.append((h * w, h, w, color))

    rectangles.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    out_h = max(h for _, h, _, _ in rectangles)
    out_w = sum(w for _, _, w, _ in rectangles) + len(rectangles) - 1
    out = np.zeros((out_h, out_w), dtype=int)

    c0 = 0
    for _area, h, w, color in rectangles:
        out[:h, c0 : c0 + w] = color
        c0 += w + 1
    return out


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = tuple(range(1, 10))
RECT_DIMS = tuple(
    (h, w)
    for h in range(3, 7)
    for w in range(3, 7)
    if 8 <= h * w <= 24
)
PAIR_COUNT_OPTIONS = (3, 4)
MAX_GENERATION_ATTEMPTS = 400
SPLIT_ATTEMPTS = 250

_RNG = np.random.default_rng()


def _components_of_color(grid: np.ndarray, color: int) -> list[set[tuple[int, int]]]:
    mask = grid == color
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: set[tuple[int, int]] = set()
            while queue:
                cr, cc = queue.popleft()
                cells.add((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(cells)
    return comps


def _normalize(cells: set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    norm = tuple(sorted((r - r0, c - c0) for r, c in cells))
    return norm, max(rows) - r0 + 1, max(cols) - c0 + 1


def _pair_rectangle_dims(
    comp_a: set[tuple[int, int]] | tuple[tuple[int, int], ...],
    comp_b: set[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> tuple[int, int] | None:
    norm_a, ha, wa = _normalize(comp_a)
    norm_b, hb, wb = _normalize(comp_b)
    set_a = set(norm_a)
    set_b = set(norm_b)

    dims: set[tuple[int, int]] = set()
    for dr in range(-hb, ha + 1):
        for dc in range(-wb, wa + 1):
            shifted_b = {(r + dr, c + dc) for r, c in set_b}
            if set_a & shifted_b:
                continue
            union = set_a | shifted_b
            rows = [r for r, _ in union]
            cols = [c for _, c in union]
            height = max(rows) - min(rows) + 1
            width = max(cols) - min(cols) + 1
            if len(union) == height * width:
                dims.add((height, width))

    if len(dims) != 1:
        return None
    return next(iter(dims))


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for dr, dc in NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen) == len(cells)


def _fragment_is_valid(cells: set[tuple[int, int]]) -> bool:
    if len(cells) < 3 or not _is_connected(cells):
        return False
    _norm, h, w = _normalize(cells)
    return h >= 2 and w >= 2 and len(cells) < h * w


def _grow_connected_subset(h: int, w: int, size: int) -> set[tuple[int, int]] | None:
    cells = {(int(_RNG.integers(h)), int(_RNG.integers(w)))}
    while len(cells) < size:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(frontier[int(_RNG.integers(len(frontier)))])
    return cells


def _mask_from_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    norm, h, w = _normalize(cells)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _generate_fragment_pair(h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    full = {(r, c) for r in range(h) for c in range(w)}
    area = h * w
    low = max(3, area // 3)
    high = min(area - 3, area - area // 3)
    if low > high:
        low, high = 3, area - 3

    for _ in range(SPLIT_ATTEMPTS):
        target = int(_RNG.integers(low, high + 1))
        part_a = _grow_connected_subset(h, w, target)
        if part_a is None:
            continue
        part_b = full - part_a
        if not _fragment_is_valid(part_a) or not _fragment_is_valid(part_b):
            continue
        dims = _pair_rectangle_dims(part_a, part_b)
        if dims != (h, w):
            continue
        return _mask_from_cells(part_a), _mask_from_cells(part_b)
    return None


def _dilate_cells(mask: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    dilated: set[tuple[int, int]] = set()
    for r, c in zip(*np.where(mask)):
        rr, cc = top + int(r), left + int(c)
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    dilated.add((nr, nc))
    return dilated


def _place_fragments(fragments: list[dict[str, object]], grid_h: int, grid_w: int) -> np.ndarray | None:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    blocked: set[tuple[int, int]] = set()

    order = list(range(len(fragments)))
    _RNG.shuffle(order)
    order.sort(key=lambda idx: (-int(np.sum(fragments[idx]["mask"])), idx))

    for idx in order:
        frag = fragments[idx]
        mask = frag["mask"]
        color = int(frag["color"])
        fh, fw = mask.shape
        positions = [(r, c) for r in range(grid_h - fh + 1) for c in range(grid_w - fw + 1)]
        _RNG.shuffle(positions)

        placed = False
        coords = [(int(r), int(c)) for r, c in zip(*np.where(mask))]
        for top, left in positions:
            if any((top + r, left + c) in blocked for r, c in coords):
                continue
            for r, c in coords:
                grid[top + r, left + c] = color
            blocked |= _dilate_cells(mask, top, left, (grid_h, grid_w))
            placed = True
            break
        if not placed:
            return None
    return grid


def _build_expected_output(specs: list[tuple[int, int, int]]) -> np.ndarray:
    ordered = sorted(((h * w, h, w, color) for color, h, w in specs), key=lambda item: (item[0], item[1], item[2], item[3]))
    out_h = max(h for _, h, _, _ in ordered)
    out_w = sum(w for _, _, w, _ in ordered) + len(ordered) - 1
    out = np.zeros((out_h, out_w), dtype=int)
    c0 = 0
    for _area, h, w, color in ordered:
        out[:h, c0 : c0 + w] = color
        c0 += w + 1
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(_RNG.choice(PAIR_COUNT_OPTIONS))
    colors = list(_RNG.choice(PALETTE, size=pair_count, replace=False))

    specs: list[tuple[int, int, int]] = []
    used_areas: set[int] = set()
    while len(specs) < pair_count:
        h, w = RECT_DIMS[int(_RNG.integers(len(RECT_DIMS)))]
        area = h * w
        if area in used_areas:
            continue
        specs.append((colors[len(specs)], h, w))
        used_areas.add(area)

    if sum(w for _, _, w in specs) + pair_count - 1 > 30:
        return None

    fragments: list[dict[str, object]] = []
    for color, h, w in specs:
        pair = _generate_fragment_pair(h, w)
        if pair is None:
            return None
        mask_a, mask_b = pair
        fragments.append({"color": color, "mask": mask_a})
        fragments.append({"color": color, "mask": mask_b})

    max_frag_h = max(int(frag["mask"].shape[0]) for frag in fragments)
    max_frag_w = max(int(frag["mask"].shape[1]) for frag in fragments)
    side_base = 12 + len(fragments)
    grid_h_low = max(max_frag_h + 4, side_base - 2)
    grid_h_high = min(30, side_base + 4)
    grid_w_low = max(max_frag_w + 4, side_base)
    grid_w_high = min(30, side_base + 6)
    if grid_h_low > grid_h_high or grid_w_low > grid_w_high:
        return None

    grid_h = int(_RNG.integers(grid_h_low, grid_h_high + 1))
    grid_w = int(_RNG.integers(grid_w_low, grid_w_high + 1))
    grid = _place_fragments(fragments, grid_h, grid_w)
    if grid is None:
        return None

    for color in colors:
        if len(_components_of_color(grid, int(color))) != 2:
            return None

    expected = _build_expected_output(specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
