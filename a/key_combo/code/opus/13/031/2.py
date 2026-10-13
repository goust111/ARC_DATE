from __future__ import annotations

import random
from collections import deque

import numpy as np


Coord = tuple[int, int]
Box = tuple[int, int, int, int]

BG = 0
TEMPLATE_COLOR = 5
LEGEND_SAFE_COLORS = (1, 2, 3, 4, 7, 8, 9)
MAX_GENERATION_RETRIES = 400
SUBSET_RETRIES = 160


TEMPLATE_LIBRARY = {
    "solid": (
        (
            ".##",
            "###",
            "#..",
        ),
        (
            "###.",
            ".###",
        ),
        (
            "##..",
            "###.",
            ".##.",
        ),
        (
            "###",
            "##.",
            ".##",
        ),
        (
            ".###",
            "###.",
            "..#.",
        ),
        (
            "###.",
            "##..",
            ".##.",
        ),
    ),
    "hole": (
        (
            "###",
            "#.#",
            "###",
        ),
        (
            "###.",
            "#.#.",
            "####",
        ),
        (
            "####",
            "#..#",
            "####",
        ),
    ),
}


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_coords(mask: np.ndarray) -> list[set[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[set[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: set[Coord] = set()
            while q:
                cr, cc = q.popleft()
                comp.add((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: set[Coord]) -> Box:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _clip_expand(box: Box, shape: tuple[int, int], pad: int = 1) -> Box:
    h, w = shape
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _boxes_overlap(a: Box, b: Box) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _mask_from_strings(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_transforms(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for base in (mask, np.fliplr(mask)):
        for k in range(4):
            transformed = _crop_mask(np.rot90(base, k))
            key = transformed.tobytes() + bytes(transformed.shape)
            if key not in seen:
                seen.add(key)
                variants.append(transformed)
    return variants


def _normalize_cells(cells: set[Coord]) -> tuple[Coord, ...]:
    r0, c0, _r1, _c1 = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _is_irregular(mask: np.ndarray) -> bool:
    return int(mask.sum()) != mask.shape[0] * mask.shape[1]


def _template_has_hole(mask: np.ndarray) -> bool:
    return bool(_interior_holes(mask))


def _random_template() -> np.ndarray:
    family = "hole" if random.random() < 0.5 else "solid"
    base = _mask_from_strings(random.choice(TEMPLATE_LIBRARY[family]))
    variants = [variant for variant in _mask_transforms(base) if 5 <= int(variant.sum()) <= 10 and _is_irregular(variant)]
    return random.choice(variants)


def _interior_holes(mask: np.ndarray) -> set[Coord]:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros_like(mask, dtype=bool)
    exterior = np.zeros_like(mask, dtype=bool)
    q = deque()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                exterior[r, c] = True
                q.append((r, c))
    while q:
        cr, cc = q.popleft()
        for nr, nc in _neighbors4(cr, cc):
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                exterior[nr, nc] = True
                q.append((nr, nc))
    holes = set()
    for r in range(h):
        for c in range(w):
            if bg[r, c] and not exterior[r, c]:
                holes.add((r, c))
    return holes


def _translated_cells(offsets: tuple[Coord, ...], top: int, left: int) -> set[Coord]:
    return {(top + dr, left + dc) for dr, dc in offsets}


def _halo_cells_from_object(cells: set[Coord], shape: tuple[int, int]) -> set[Coord]:
    r0, c0, r1, c1 = _bbox(cells)
    pr0, pc0, pr1, pc1 = _clip_expand((r0, c0, r1, c1), shape, pad=1)
    local_h = pr1 - pr0 + 1
    local_w = pc1 - pc0 + 1
    local_mask = np.zeros((local_h, local_w), dtype=bool)
    for r, c in cells:
        local_mask[r - pr0, c - pc0] = True
    holes = _interior_holes(local_mask)
    halo: set[Coord] = set()
    for rr in range(local_h):
        for cc in range(local_w):
            if local_mask[rr, cc] or (rr, cc) in holes:
                continue
            halo.add((pr0 + rr, pc0 + cc))
    return halo


def _legend_mapping(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    assert w >= 2
    mapping: dict[int, int] = {}
    legend_mask = np.zeros_like(grid, dtype=bool)
    r = 0
    while r <= h - 2:
        block = grid[r : r + 2, 0:2]
        if np.all(block != BG) and np.all(block[:, 0] == block[0, 0]) and np.all(block[:, 1] == block[0, 1]):
            src = int(block[0, 0])
            halo = int(block[0, 1])
            assert src != halo
            assert src not in mapping
            mapping[src] = halo
            legend_mask[r : r + 2, 0:2] = True
            r += 2
            continue
        r += 1
    assert 2 <= len(mapping) <= 3
    return mapping, legend_mask


def _template_offsets(grid: np.ndarray) -> tuple[tuple[Coord, ...], set[Coord]]:
    comps = _component_coords(grid == TEMPLATE_COLOR)
    assert len(comps) == 1
    cells = comps[0]
    offsets = _normalize_cells(cells)
    return offsets, cells


def _candidate_placements(grid: np.ndarray, color: int, fragment_cells: set[Coord], offsets: tuple[Coord, ...]) -> list[set[Coord]]:
    h, w = grid.shape
    candidates: list[set[Coord]] = []
    seen: set[tuple[int, int]] = set()
    for pr, pc in fragment_cells:
        for dr, dc in offsets:
            top = pr - dr
            left = pc - dc
            if (top, left) in seen:
                continue
            seen.add((top, left))
            cells = _translated_cells(offsets, top, left)
            if not all(0 <= r < h and 0 <= c < w for r, c in cells):
                continue
            if not fragment_cells <= cells:
                continue
            valid = True
            for r, c in cells:
                val = int(grid[r, c])
                if (r, c) in fragment_cells:
                    if val != color:
                        valid = False
                        break
                elif val != BG:
                    valid = False
                    break
            if valid:
                candidates.append(cells)
    return candidates


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    mapping, legend_mask = _legend_mapping(g)
    offsets, template_cells = _template_offsets(g)

    completed: dict[int, set[Coord]] = {}
    for color in sorted(mapping):
        fragment_cells = {
            (int(r), int(c))
            for r, c in np.argwhere((g == color) & (~legend_mask))
        }
        assert len(fragment_cells) >= 2
        candidates = _candidate_placements(g, color, fragment_cells, offsets)
        unique = {frozenset(cells) for cells in candidates}
        assert len(unique) == 1, f"fragment color {color} must determine a unique translated copy"
        completed[color] = set(next(iter(unique)))

    out = g.copy()
    for color, cells in completed.items():
        for r, c in cells:
            out[r, c] = color

    halo_claimed: set[Coord] = set()
    for color, cells in completed.items():
        halo_color = mapping[color]
        halo_cells = _halo_cells_from_object(cells, out.shape)
        for r, c in halo_cells:
            if out[r, c] != BG:
                continue
            assert (r, c) not in halo_claimed
            out[r, c] = halo_color
            halo_claimed.add((r, c))

    for r, c in template_cells:
        out[r, c] = TEMPLATE_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each colored fragment into the gray template's translated copy, then draw a legend-colored halo around the completed shape while preserving enclosed holes.

    Input structure:
    - The background is black, with 2x2 color-legend blocks stacked on the left edge.
    - A single gray 4-connected template appears elsewhere as the only complete shape of that color.
    - Each legend source color also appears as a partial translated copy of the gray template, with missing pixels removed but no overlap between completed copies.

    Rule:
    - Read each 2x2 left-edge legend block as source color in its left column and halo color in its right column.
    - Use the gray component as the template mask; for every non-gray fragment color, find the unique translation of that mask that contains all observed pixels of that color, then fill in the missing template cells with the fragment color.
    - For each completed copy, expand its tight bounding box by one cell in every direction, clipped to the grid, and recolor only black background cells in that region to the legend's halo color, except for enclosed black holes inside the completed copy, which stay black.

    Key insight:
    - The fragment pixels only anchor where the full template copy must be completed; that completed geometry, not the fragment alone, determines the halo region, while the left-edge legend determines the halo color.
    """
    return _solve(grid)


def _legend_rows(height: int, count: int) -> list[int]:
    gap_choices = [random.randint(1, 2) for _ in range(count - 1)]
    total = 2 * count + sum(gap_choices)
    start = random.randint(0, height - total)
    rows = [start]
    cur = start
    for gap in gap_choices:
        cur += 2 + gap
        rows.append(cur)
    return rows


def _place_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> set[Coord]:
    cells: set[Coord] = set()
    for r, c in np.argwhere(mask):
        rr = top + int(r)
        cc = left + int(c)
        grid[rr, cc] = color
        cells.add((rr, cc))
    return cells


def _reserve_box(top: int, left: int, mask: np.ndarray, shape: tuple[int, int]) -> Box:
    h, w = mask.shape
    return _clip_expand((top, left, top + h - 1, left + w - 1), shape, pad=1)


def _positions_for_copy(height: int, width: int, mask: np.ndarray, central_only: bool = False) -> list[tuple[int, int]]:
    h, w = mask.shape
    positions = []
    row_values = list(range(0, height - h + 1))
    col_min = max(4, width // 2) if central_only else 4
    col_values = list(range(col_min, width - w + 1))
    for top in row_values:
        for left in col_values:
            positions.append((top, left))
    random.shuffle(positions)
    return positions


def _select_subset(full_cells: set[Coord], grid_shape: tuple[int, int]) -> set[Coord]:
    cells = sorted(full_cells)
    full_halo = _halo_cells_from_object(full_cells, grid_shape)
    for _ in range(SUBSET_RETRIES):
        keep_n = random.randint(2, len(cells) - 1)
        subset = set(random.sample(cells, keep_n))
        if _halo_cells_from_object(subset, grid_shape) == full_halo:
            continue
        return subset

    boundary = set()
    r0, c0, r1, c1 = _bbox(full_cells)
    for cell in full_cells:
        if cell[0] in (r0, r1) or cell[1] in (c0, c1):
            boundary.add(cell)
    core = sorted(full_cells - boundary)
    boundary = sorted(boundary)
    remove_pool = boundary if boundary else sorted(full_cells)
    for cell in remove_pool:
        subset = set(full_cells)
        subset.remove(cell)
        if len(subset) >= 2 and _halo_cells_from_object(subset, grid_shape) != full_halo:
            return subset
    raise AssertionError("could not choose a non-degenerate fragment subset")


def _cells_to_mask(cells: set[Coord]) -> np.ndarray:
    if not cells:
        return np.zeros((1, 1), dtype=bool)
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _build_case() -> tuple[np.ndarray, np.ndarray, dict]:
    height = random.randint(18, 24)
    width = random.randint(20, 28)
    count = random.randint(2, 3)
    colors = random.sample(LEGEND_SAFE_COLORS, 2 * count)
    src_colors = colors[:count]
    halo_colors = colors[count:]

    template_mask = _random_template()
    grid_output = np.zeros((height, width), dtype=int)

    legend_rows = _legend_rows(height, count)
    legend_mask = np.zeros_like(grid_output, dtype=bool)
    mapping: dict[int, int] = {}
    for top, src, halo in zip(legend_rows, src_colors, halo_colors, strict=True):
        grid_output[top : top + 2, 0] = src
        grid_output[top : top + 2, 1] = halo
        legend_mask[top : top + 2, 0:2] = True
        mapping[src] = halo

    reserved: list[Box] = []
    template_top = None
    template_left = None
    for top, left in _positions_for_copy(height, width, template_mask, central_only=True):
        box = _reserve_box(top, left, template_mask, grid_output.shape)
        if any(_boxes_overlap(box, other) for other in reserved):
            continue
        template_top, template_left = top, left
        reserved.append(box)
        break
    assert template_top is not None and template_left is not None
    template_cells = _place_mask(grid_output, template_mask, template_top, template_left, TEMPLATE_COLOR)
    template_holes = {
        (template_top + r, template_left + c) for r, c in _interior_holes(template_mask)
    }

    copies: dict[int, set[Coord]] = {}
    reserve_boxes: dict[int, Box] = {}
    for color in src_colors:
        placed = False
        for top, left in _positions_for_copy(height, width, template_mask, central_only=False):
            box = _reserve_box(top, left, template_mask, grid_output.shape)
            if any(_boxes_overlap(box, other) for other in reserved):
                continue
            cells = _translated_cells(_normalize_cells(template_cells), top, left)
            if cells & template_cells:
                continue
            copies[color] = cells
            reserve_boxes[color] = box
            reserved.append(box)
            placed = True
            break
        if not placed:
            raise AssertionError("failed to place all copies")

    for color, cells in copies.items():
        for r, c in cells:
            grid_output[r, c] = color

    halo_cells_by_color: dict[int, set[Coord]] = {}
    claimed_halo: set[Coord] = set()
    for color, cells in copies.items():
        halo_cells = _halo_cells_from_object(cells, grid_output.shape)
        assert halo_cells.isdisjoint(template_cells)
        assert halo_cells.isdisjoint(claimed_halo)
        halo_cells_by_color[color] = halo_cells
        claimed_halo |= halo_cells
        for r, c in halo_cells:
            if grid_output[r, c] == BG:
                grid_output[r, c] = mapping[color]

    grid_input = grid_output.copy()
    for cells in halo_cells_by_color.values():
        for r, c in cells:
            if grid_input[r, c] != TEMPLATE_COLOR:
                grid_input[r, c] = BG

    fragments: dict[int, set[Coord]] = {}
    for color, cells in copies.items():
        subset = _select_subset(cells, grid_output.shape)
        fragments[color] = subset
        for r, c in cells - subset:
            grid_input[r, c] = BG

    meta = {
        "mapping": mapping,
        "legend_mask": legend_mask,
        "template_cells": template_cells,
        "template_holes": template_holes,
        "copies": copies,
        "fragments": fragments,
        "halo_cells": halo_cells_by_color,
        "reserve_boxes": reserve_boxes,
    }
    return grid_input, grid_output, meta


def _validate_case(grid_input: np.ndarray, grid_output: np.ndarray, meta: dict) -> bool:
    try:
        solved = main(grid_input)
    except AssertionError:
        return False
    if not np.array_equal(solved, grid_output):
        return False

    mapping, legend_mask = _legend_mapping(grid_input)
    if mapping != meta["mapping"]:
        return False

    offsets, _template_cells = _template_offsets(grid_input)
    halo_union = set()
    naive_halo_diff = False

    for color, expected_full in meta["copies"].items():
        fragment_cells = meta["fragments"][color]
        candidates = _candidate_placements(grid_input, color, fragment_cells, offsets)
        unique = {frozenset(cells) for cells in candidates}
        if unique != {frozenset(expected_full)}:
            return False

        full_halo = _halo_cells_from_object(expected_full, grid_input.shape)
        fragment_halo = _halo_cells_from_object(fragment_cells, grid_input.shape)
        if fragment_halo == full_halo:
            return False
        naive_halo_diff = True

        if full_halo & halo_union:
            return False
        halo_union |= full_halo

        for r, c in expected_full:
            if grid_output[r, c] != color:
                return False
        for r, c in full_halo:
            if grid_output[r, c] == BG:
                return False

    if not naive_halo_diff:
        return False

    template_holes = meta["template_holes"]
    if template_holes:
        template_box = _bbox(meta["template_cells"])
        template_r0, template_c0, _template_r1, _template_c1 = template_box
        hole_offsets = {(r - template_r0, c - template_c0) for r, c in template_holes}
        for expected_full in meta["copies"].values():
            copy_r0, copy_c0, _copy_r1, _copy_c1 = _bbox(expected_full)
            for dr, dc in hole_offsets:
                hole_cell = (copy_r0 + dr, copy_c0 + dc)
                if grid_output[hole_cell] != BG:
                    return False

    if len({frozenset(cells) for cells in meta["copies"].values()}) != len(meta["copies"]):
        return False

    if len({tuple(sorted(cells)) for cells in meta["fragments"].values()}) != len(meta["fragments"]):
        return False

    if np.array_equal(grid_input, grid_output):
        return False

    # Legend colors must not leak into puzzle objects.
    for color in mapping:
        fragment_pixels = {
            (int(r), int(c))
            for r, c in np.argwhere((grid_input == color) & (~legend_mask))
        }
        if len(fragment_pixels) < 2:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_RETRIES):
        try:
            grid_input, grid_output, meta = _build_case()
        except AssertionError:
            continue
        if _validate_case(grid_input, grid_output, meta):
            return grid_input
    raise RuntimeError("failed to generate a valid puzzle instance")
