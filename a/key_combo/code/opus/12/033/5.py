from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
INPUT_COLORS = (1, 3, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 400
LOCAL_ATTEMPTS = 200


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _mirror_coords(coords: Iterable[tuple[int, int]], height: int) -> frozenset[tuple[int, int]]:
    return frozenset((height - 1 - int(r), int(c)) for r, c in coords)


def _normalize(coords: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = {(int(r), int(c)) for r, c in coords}
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


def _bbox_dims(coords: Iterable[tuple[int, int]]) -> tuple[int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _translate(coords: Iterable[tuple[int, int]], dr: int, dc: int) -> frozenset[tuple[int, int]]:
    return frozenset((int(r) + int(dr), int(c) + int(dc)) for r, c in coords)


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    seen = np.zeros(g.shape, dtype=bool)
    components: list[dict] = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color == BLACK or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(g.shape, nr, nc) and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coord_set = frozenset(coords)
            components.append(
                {
                    "color": color,
                    "coords": tuple(sorted(coords)),
                    "coord_set": coord_set,
                    "area": len(coord_set),
                }
            )
    return components


def _paired_component_ids(components: list[dict], height: int) -> set[int]:
    lookup = {(comp["color"], comp["coord_set"]): idx for idx, comp in enumerate(components)}
    paired: set[int] = set()
    for idx, comp in enumerate(components):
        partner = lookup.get((comp["color"], _mirror_coords(comp["coord_set"], height)))
        if partner is not None and partner != idx:
            paired.add(idx)
            paired.add(partner)
    return paired


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify horizontally mirrored same-color components, recolor them, and mirror only the unmatched ones.

    Input structure:
    - The grid has a black background and several monochrome 4-connected foreground components.
    - Some components come in same-color horizontal mirror pairs across the grid's central horizontal axis.
    - The remaining components have no exact same-color mirrored partner.

    Rule:
    - Use the grid height to define the horizontal reflection axis and mirror each component across it.
    - If another component of the same color occupies exactly that mirrored coordinate set, mark both as a symmetric pair.
    - Recolor every paired component to yellow.
    - Recolor every unpaired component to red and also paint its reflected copy in red.
    - If a reflected red copy lands on a yellow pixel, keep the yellow pixel.

    Key insight:
    - Symmetry is not used to repair a missing pattern here; it is used first as a classifier, and that classification controls whether a component is only recolored or is both recolored and mirrored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    height, width = map(int, g.shape)
    components = _find_components(g)
    paired = _paired_component_ids(components, height)

    out = np.zeros((height, width), dtype=int)
    for idx, comp in enumerate(components):
        if idx in paired:
            for r, c in comp["coords"]:
                out[r, c] = YELLOW

    for idx, comp in enumerate(components):
        if idx in paired:
            continue
        for r, c in comp["coords"]:
            out[r, c] = RED
        for r, c in _mirror_coords(comp["coord_set"], height):
            if out[r, c] != YELLOW:
                out[r, c] = RED

    return out


def _random_shape(target_area: int) -> tuple[tuple[int, int], ...]:
    for _ in range(LOCAL_ATTEMPTS):
        cells = {(0, 0)}
        while len(cells) < target_area:
            frontier: list[tuple[tuple[int, int], ...]] = []
            for r, c in list(cells):
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in cells:
                        continue
                    candidate = _normalize(cells | {(nr, nc)})
                    h, w = _bbox_dims(candidate)
                    if h <= 4 and w <= 5:
                        frontier.append(candidate)
            if not frontier:
                break
            cells = set(random.choice(frontier))
        if len(cells) != target_area:
            continue
        norm = _normalize(cells)
        h, w = _bbox_dims(norm)
        box_area = h * w
        if target_area >= 5 and box_area == target_area and random.random() < 0.8:
            continue
        return norm
    raise RuntimeError("failed to build a polyomino")


def _coords_clear(coords: Iterable[tuple[int, int]], occ_mask: np.ndarray) -> bool:
    h, w = occ_mask.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        if occ_mask[r0:r1, c0:c1].any():
            return False
    return True


def _reserve(coords: Iterable[tuple[int, int]], mask: np.ndarray) -> None:
    for r, c in coords:
        mask[r, c] = True


def _safe_top_rows(height: int) -> int:
    return (height - 1) // 2


def _place_pair(
    height: int,
    width: int,
    shape: tuple[tuple[int, int], ...],
    input_occ: np.ndarray,
    output_occ: np.ndarray,
) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]] | None:
    top_rows = _safe_top_rows(height)
    sh, sw = _bbox_dims(shape)
    if sh > top_rows or sw > width:
        return None
    for _ in range(LOCAL_ATTEMPTS):
        r0 = random.randint(0, top_rows - sh)
        c0 = random.randint(0, width - sw)
        top = _translate(shape, r0, c0)
        bottom = _mirror_coords(top, height)
        full = top | bottom
        if len(full) != 2 * len(top):
            continue
        if not _coords_clear(full, input_occ):
            continue
        if any(output_occ[r, c] for r, c in full):
            continue
        return top, bottom
    return None


def _place_unpaired(
    height: int,
    width: int,
    shape: tuple[tuple[int, int], ...],
    input_occ: np.ndarray,
    output_occ: np.ndarray,
) -> frozenset[tuple[int, int]] | None:
    top_rows = _safe_top_rows(height)
    sh, sw = _bbox_dims(shape)
    if sh > top_rows or sw > width:
        return None

    for _ in range(LOCAL_ATTEMPTS):
        r0 = random.randint(0, top_rows - sh)
        c0 = random.randint(0, width - sw)
        base_top = _translate(shape, r0, c0)
        if random.random() < 0.5:
            coords = base_top
        else:
            coords = _mirror_coords(base_top, height)
        reflected = _mirror_coords(coords, height)
        future = coords | reflected
        if not _coords_clear(coords, input_occ):
            continue
        if any(output_occ[r, c] for r, c in future):
            continue
        return coords
    return None


def _render_components(shape: tuple[int, int], placed: list[tuple[int, frozenset[tuple[int, int]]]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for color, coords in placed:
        for r, c in coords:
            grid[r, c] = color
    return grid


def _independent_expected(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    comps = _find_components(g)
    paired = _paired_component_ids(comps, g.shape[0])
    expected = np.zeros_like(g)
    for idx, comp in enumerate(comps):
        if idx in paired:
            for r, c in comp["coords"]:
                expected[r, c] = YELLOW
    for idx, comp in enumerate(comps):
        if idx in paired:
            continue
        for r, c in comp["coords"]:
            expected[r, c] = RED
        for r, c in _mirror_coords(comp["coord_set"], g.shape[0]):
            if expected[r, c] != YELLOW:
                expected[r, c] = RED
    return expected


def _valid_input(grid: np.ndarray) -> bool:
    components = _find_components(grid)
    if not 4 <= len(components) <= 6:
        return False
    if any(not (4 <= comp["area"] <= 10) for comp in components):
        return False

    paired = _paired_component_ids(components, grid.shape[0])
    if len(paired) < 2 or len(paired) == len(components):
        return False

    # Keep input visually clear: no unpaired mirror location is already occupied in the input.
    occupied = {coord for comp in components for coord in comp["coord_set"]}
    for idx, comp in enumerate(components):
        if idx in paired:
            continue
        if _mirror_coords(comp["coord_set"], grid.shape[0]) & occupied:
            return False

    expected = _independent_expected(grid)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if not np.any(actual == YELLOW) or not np.any(actual == RED):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(13, 18)
    width = random.randint(12, 18)

    layout_options = [(1, 2), (1, 3), (1, 4), (2, 1), (2, 2)]
    n_pairs, n_unpaired = random.choice(layout_options)
    unit_count = n_pairs + n_unpaired
    colors = random.sample(INPUT_COLORS, unit_count)

    input_occ = np.zeros((height, width), dtype=bool)
    output_occ = np.zeros((height, width), dtype=bool)
    placed: list[tuple[int, frozenset[tuple[int, int]]]] = []
    color_idx = 0

    for _ in range(n_pairs):
        shape = _random_shape(random.randint(4, 9))
        placement = _place_pair(height, width, shape, input_occ, output_occ)
        if placement is None:
            return None
        top, bottom = placement
        color = colors[color_idx]
        color_idx += 1
        placed.append((color, top))
        placed.append((color, bottom))
        _reserve(top | bottom, input_occ)
        _reserve(top | bottom, output_occ)

    for _ in range(n_unpaired):
        shape = _random_shape(random.randint(4, 10))
        coords = _place_unpaired(height, width, shape, input_occ, output_occ)
        if coords is None:
            return None
        color = colors[color_idx]
        color_idx += 1
        placed.append((color, coords))
        _reserve(coords, input_occ)
        _reserve(coords | _mirror_coords(coords, height), output_occ)

    grid = _render_components((height, width), placed)
    if _valid_input(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
