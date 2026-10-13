from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
SEED_COLOR = 7  # orange
RING_PALETTE = (1, 2, 3, 4, 5, 6, 8, 9)

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
STRUCTURE_COUNT_CHOICES = (1, 2, 3)
RING_COUNT_CHOICES = (2, 3)
PLACEMENT_GAP = 1
GEN_RETRIES = 400

_SEED_MASKS = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _ring_boundary(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords = [(r0, c) for c in range(c0, c1 + 1)]
    if r1 != r0:
        coords.extend((r1, c) for c in range(c0, c1 + 1))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        if c1 != c0:
            coords.append((r, c1))
    return coords


def _draw_ring(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    for r, c in _ring_boundary(r0, c0, r1, c1):
        grid[r, c] = color


def _grow_layers(
    shape: tuple[int, int],
    seed_cells: list[tuple[int, int]],
    colors_inner_to_outer: tuple[int, ...],
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BG, dtype=int)
    occupied = set(seed_cells)
    for r, c in seed_cells:
        out[r, c] = SEED_COLOR
    for color in colors_inner_to_outer:
        frontier: set[tuple[int, int]] = set()
        for r, c in occupied:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in occupied:
                    frontier.add((nr, nc))
        for r, c in frontier:
            out[r, c] = color
        occupied |= frontier
    return out


def _parse_structure(
    grid: np.ndarray, bbox: tuple[int, int, int, int]
) -> dict[str, object] | None:
    r0, c0, r1, c1 = bbox
    max_layers = min((r1 - r0 + 1) // 2, (c1 - c0 + 1) // 2)
    outer_to_inner: list[int] = []
    ring_cells: list[tuple[int, int]] = []

    for depth in range(max_layers):
        boundary = _ring_boundary(r0 + depth, c0 + depth, r1 - depth, c1 - depth)
        values = {int(grid[r, c]) for r, c in boundary}
        if len(values) != 1:
            break
        color = next(iter(values))
        if color in (BG, SEED_COLOR):
            break
        outer_to_inner.append(color)
        ring_cells.extend(boundary)

    if len(outer_to_inner) not in RING_COUNT_CHOICES:
        return None
    if len(set(outer_to_inner)) != len(outer_to_inner):
        return None

    ring_count = len(outer_to_inner)
    cr0, cc0 = r0 + ring_count, c0 + ring_count
    cr1, cc1 = r1 - ring_count, c1 - ring_count
    if cr0 > cr1 or cc0 > cc1:
        return None

    seed_cells: list[tuple[int, int]] = []
    for r in range(cr0, cr1 + 1):
        for c in range(cc0, cc1 + 1):
            value = int(grid[r, c])
            if value == SEED_COLOR:
                seed_cells.append((r, c))
            elif value != BG:
                return None
    if not (1 <= len(seed_cells) <= 3):
        return None

    local_seed = np.zeros((cr1 - cr0 + 1, cc1 - cc0 + 1), dtype=bool)
    for r, c in seed_cells:
        local_seed[r - cr0, c - cc0] = True
    if len(_components(local_seed)) != 1:
        return None

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            depth = min(r - r0, r1 - r, c - c0, c1 - c)
            value = int(grid[r, c])
            if depth < ring_count:
                if value != outer_to_inner[depth]:
                    return None
            elif value not in (BG, SEED_COLOR):
                return None

    return {
        "bbox": bbox,
        "ring_cells": ring_cells,
        "seed_cells": seed_cells,
        "colors_inner_to_outer": tuple(reversed(outer_to_inner)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each concentric ring structure into outward growth layers around its orange seed.

    Input structure:
    - The background is black.
    - One or more independent structures contain 2-3 nested one-cell-thick rectangular rings with different colors from outer to inner.
    - Inside each innermost ring is a small orange seed pattern occupying 1-3 cells on the black cavity.

    Rule:
    - For each structure, read the ring colors from inner to outer and keep only the orange seed.
    - Remove all ring pixels back to black.
    - Starting from the seed, perform one 4-neighbor dilation per recorded ring layer; the newly added cells of step k take the k-th inner-to-outer ring color, while the seed itself stays orange.

    Key insight:
    - The nested colored rectangles are only an instruction code: their colors specify the successive dilation layers, while the seed shape determines the grown silhouette.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()

    structures: list[dict[str, object]] = []
    seen_boxes: set[tuple[int, int, int, int]] = set()
    for comp in _components(g != BG):
        if all(int(g[r, c]) == SEED_COLOR for r, c in comp):
            continue
        box = _bbox(comp)
        if box in seen_boxes:
            continue
        parsed = _parse_structure(g, box)
        if parsed is None:
            continue
        seen_boxes.add(box)
        structures.append(parsed)

    assert structures, "expected at least one concentric ring structure"

    for structure in structures:
        for r, c in structure["ring_cells"]:
            out[r, c] = BG

    h, w = g.shape
    for structure in structures:
        occupied = set(structure["seed_cells"])
        for color in structure["colors_inner_to_outer"]:
            frontier: set[tuple[int, int]] = set()
            for r, c in occupied:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in occupied:
                        frontier.add((nr, nc))
            for r, c in frontier:
                assert out[r, c] in (BG, color), "grown structures must stay disjoint"
                out[r, c] = color
            occupied |= frontier

    return out


def _seed_cells_from_mask(seed_mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r, c in np.argwhere(seed_mask):
        cells.append((int(top + r), int(left + c)))
    return cells


def _make_structure_spec() -> dict[str, object]:
    ring_count = random.choice(RING_COUNT_CHOICES)
    seed_mask = random.choice(_SEED_MASKS)
    sh, sw = seed_mask.shape

    top_margin = random.randint(0, 2)
    bottom_margin = random.randint(0, 2)
    left_margin = random.randint(0, 2)
    right_margin = random.randint(0, 2)
    cavity_h = sh + top_margin + bottom_margin
    cavity_w = sw + left_margin + right_margin

    patch_h = cavity_h + 2 * ring_count
    patch_w = cavity_w + 2 * ring_count
    input_patch = np.full((patch_h, patch_w), BG, dtype=int)

    colors_inner_to_outer = tuple(random.sample(RING_PALETTE, ring_count))
    colors_outer_to_inner = tuple(reversed(colors_inner_to_outer))
    for depth, color in enumerate(colors_outer_to_inner):
        _draw_ring(input_patch, depth, depth, patch_h - 1 - depth, patch_w - 1 - depth, color)

    seed_top = ring_count + top_margin
    seed_left = ring_count + left_margin
    input_patch[seed_top : seed_top + sh, seed_left : seed_left + sw][seed_mask] = SEED_COLOR
    seed_cells = _seed_cells_from_mask(seed_mask, seed_top, seed_left)
    output_patch = _grow_layers(input_patch.shape, seed_cells, colors_inner_to_outer)

    return {
        "input_patch": input_patch,
        "output_patch": output_patch,
        "shape": input_patch.shape,
        "area": patch_h * patch_w,
    }


def _boxes_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0
) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_structures(
    specs: list[dict[str, object]], h: int, w: int
) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    occupied_boxes: list[tuple[int, int, int, int]] = []

    for spec in sorted(specs, key=lambda item: int(item["area"]), reverse=True):
        ph, pw = spec["shape"]
        candidates: list[tuple[int, int]] = []
        for r0 in range(h - ph + 1):
            for c0 in range(w - pw + 1):
                box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if any(_boxes_overlap(box, other, gap=PLACEMENT_GAP) for other in occupied_boxes):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        ph, pw = spec["shape"]
        assert np.all(inp[r0 : r0 + ph, c0 : c0 + pw] == BG)
        assert np.all(expected[r0 : r0 + ph, c0 : c0 + pw] == BG)
        inp[r0 : r0 + ph, c0 : c0 + pw] = spec["input_patch"]
        expected[r0 : r0 + ph, c0 : c0 + pw] = spec["output_patch"]
        occupied_boxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))

    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    structure_count = random.choice(STRUCTURE_COUNT_CHOICES)
    specs = [_make_structure_spec() for _ in range(structure_count)]

    min_h = max(int(spec["shape"][0]) for spec in specs) + 2
    min_w = max(int(spec["shape"][1]) for spec in specs) + 2
    h = random.randint(max(GRID_H_RANGE[0], min_h), GRID_H_RANGE[1])
    w = random.randint(max(GRID_W_RANGE[0], min_w), GRID_W_RANGE[1])

    placed = _place_structures(specs, h, w)
    if placed is None:
        return None
    inp, expected = placed

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-growth ARC input")
