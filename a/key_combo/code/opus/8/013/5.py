from __future__ import annotations

import random
from collections import deque

import numpy as np


STACK_ORDER = [0, 3, 2, 1]  # TL -> BR -> BL -> TR
ALT_STACK_ORDER = [0, 1, 2, 3]
BASE_COLORS = [1, 2, 3, 4]
MARKER_COLOR = 9
BG = 0


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(mask.sum())


def _quadrants(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square grid"
    assert g.shape[0] % 2 == 0, "expected an even side length"
    half = g.shape[0] // 2
    return [
        g[:half, :half].copy(),
        g[:half, half:].copy(),
        g[half:, :half].copy(),
        g[half:, half:].copy(),
    ]


def _marker_rotation(patch: np.ndarray) -> int | None:
    coords = np.argwhere(patch == MARKER_COLOR)
    if coords.size == 0:
        return None
    assert len(coords) == 3, "marker must have exactly three cells"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert int(r1 - r0) == 1 and int(c1 - c0) == 1, "marker must occupy a 2x2 bbox"
    norm = {tuple(map(int, xy - np.array([r0, c0]))) for xy in coords}
    assert len(norm) == 3, "marker cells must be distinct"
    missing = next(cell for cell in ((0, 0), (0, 1), (1, 0), (1, 1)) if cell not in norm)
    return {
        (1, 1): 3,  # bend at upper-left -> CW 90
        (1, 0): 1,  # bend at upper-right -> CCW 90
        (0, 1): 2,  # bend at lower-left -> 180
        (0, 0): 0,  # bend at lower-right -> 0
    }[missing]


def _extract_mask(patch: np.ndarray) -> tuple[np.ndarray, int]:
    colors = sorted({int(v) for v in patch.ravel()} - {BG, MARKER_COLOR})
    assert len(colors) == 1, "each quadrant must contain exactly one foreground color"
    color = colors[0]
    mask = patch == color
    assert 3 <= int(mask.sum()) <= 8, "quadrant mask area must be 3..8"
    assert _is_connected(mask), "quadrant mask must be 4-connected"
    return mask, color


def _overlay(layers: list[np.ndarray], order: list[int]) -> np.ndarray:
    out = np.zeros_like(layers[0])
    for idx in order:
        patch = layers[idx]
        nz = patch != BG
        out[nz] = patch[nz]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each quadrant according to its L marker, erase the markers, then fold the four quadrants onto one shared canvas.

    Input structure:
    - An even-sized square grid is split by its horizontal and vertical midlines into four equal quadrants.
    - Each quadrant contains one 4-connected monochrome mask in one of four distinct foreground colors.
    - Some quadrants also contain a maroon 3-cell L marker whose bend orientation encodes a rotation for that quadrant.

    Rule:
    - In each quadrant, detect whether the maroon L marker is present and read its bend orientation.
    - Upper-right bend means rotate that whole quadrant counterclockwise 90 degrees; upper-left means clockwise 90; lower-left means 180; lower-right means 0 degrees.
    - Erase every maroon marker cell to background before folding.
    - Fold the four processed quadrants onto one quadrant-sized canvas with transparent overlay order: top-left, bottom-right, bottom-left, top-right; later layers overwrite earlier ones.

    Key insight:
    - The L marker is a per-quadrant control code: it chooses a local rotation before folding, so both the final overlap pattern and the visible top color depend on reading each marker correctly.
    """
    quadrants = _quadrants(np.asarray(grid, dtype=int))
    colors_seen: list[int] = []
    layers: list[np.ndarray] = []
    for patch in quadrants:
        rotation = _marker_rotation(patch)
        mask, color = _extract_mask(patch)
        colors_seen.append(color)
        cleaned = np.zeros_like(patch)
        cleaned[mask] = color
        if rotation is not None:
            cleaned = np.rot90(cleaned, rotation)
        layers.append(cleaned)
    assert len(set(colors_seen)) == 4, "quadrant colors must be distinct"
    return _overlay(layers, STACK_ORDER)


def _marker_cells(size: int, rotation: int | None) -> set[tuple[int, int]]:
    if rotation is None:
        return set()
    last = size - 1
    if rotation == 3:
        return {(0, 0), (0, 1), (1, 0)}
    if rotation == 1:
        return {(0, last), (0, last - 1), (1, last)}
    if rotation == 2:
        return {(last, 0), (last - 1, 0), (last, 1)}
    assert rotation == 0
    return {(last, last), (last - 1, last), (last, last - 1)}


def _mask_from_cells(size: int, cells: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _cells_from_mask(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, xy)) for xy in np.argwhere(mask)}


def _grow_shape(
    size: int,
    area: int,
    *,
    required: set[tuple[int, int]] | None = None,
    forbidden: set[tuple[int, int]] | None = None,
) -> np.ndarray | None:
    required = set() if required is None else set(required)
    forbidden = set() if forbidden is None else set(forbidden)
    if required & forbidden or len(required) > area:
        return None
    all_cells = {(r, c) for r in range(size) for c in range(size)}
    available = all_cells - forbidden
    if not available:
        return None
    if not required:
        start = random.choice(tuple(available))
        required.add(start)
    if not _is_connected(_mask_from_cells(size, required)):
        return None
    cells = set(required)
    frontier = {
        nbr
        for r, c in cells
        for nbr in _neighbors4(r, c)
        if 0 <= nbr[0] < size and 0 <= nbr[1] < size and nbr in available and nbr not in cells
    }
    while len(cells) < area and frontier:
        choices = sorted(frontier)
        weights = []
        for r, c in choices:
            support = sum((rr, cc) in cells for rr, cc in _neighbors4(r, c))
            weights.append((support + 1) ** 2)
        picked = random.choices(choices, weights=weights, k=1)[0]
        frontier.remove(picked)
        cells.add(picked)
        for nbr in _neighbors4(*picked):
            if 0 <= nbr[0] < size and 0 <= nbr[1] < size and nbr in available and nbr not in cells:
                frontier.add(nbr)
    if len(cells) != area:
        return None
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if len(set(rows)) == 1 or len(set(cols)) == 1:
        return None
    bbox_area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
    if bbox_area == len(cells):
        return None
    return _mask_from_cells(size, cells)


def _mask_to_colored(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _render_input(local_masks: list[np.ndarray], colors: list[int], rotations: list[int | None]) -> np.ndarray:
    size = local_masks[0].shape[0]
    quadrants: list[np.ndarray] = []
    for mask, color, rotation in zip(local_masks, colors, rotations):
        patch = _mask_to_colored(mask, color)
        for r, c in _marker_cells(size, rotation):
            assert not bool(mask[r, c]), "marker must not overlap the quadrant mask"
            patch[r, c] = MARKER_COLOR
        quadrants.append(patch)
    top = np.hstack([quadrants[0], quadrants[1]])
    bottom = np.hstack([quadrants[2], quadrants[3]])
    return np.vstack([top, bottom])


def _overlay_bool_masks(masks: list[np.ndarray], colors: list[int], order: list[int]) -> np.ndarray:
    return _overlay([_mask_to_colored(mask, color) for mask, color in zip(masks, colors)], order)


def _with_markers_then_rotate(local_masks: list[np.ndarray], colors: list[int], rotations: list[int | None]) -> np.ndarray:
    size = local_masks[0].shape[0]
    layers: list[np.ndarray] = []
    for mask, color, rotation in zip(local_masks, colors, rotations):
        patch = _mask_to_colored(mask, color)
        for r, c in _marker_cells(size, rotation):
            patch[r, c] = MARKER_COLOR
        layers.append(np.rot90(patch, 0 if rotation is None else rotation))
    return _overlay(layers, STACK_ORDER)


def _sample_rotations() -> list[int | None]:
    marked = set(random.sample(range(4), random.randint(1, 3)))
    if not (marked & {1, 3}):
        marked.add(random.choice((1, 3)))
    rotations: list[int | None] = [None, None, None, None]
    for idx in marked:
        rotations[idx] = random.choice([0, 1, 2, 3])
    if all(rot in (None, 0) for rot in rotations):
        idx = random.choice(tuple(marked))
        rotations[idx] = random.choice([1, 2, 3])
    if all(rotations[idx] in (None, 0) for idx in (1, 3)):
        idx = random.choice([i for i in (1, 3) if rotations[i] is not None])
        rotations[idx] = random.choice([1, 2, 3])
    return rotations


def _sample_scene() -> np.ndarray | None:
    size = random.randint(5, 8)
    colors = random.sample(BASE_COLORS, 4)
    rotations = _sample_rotations()
    shared = (random.randint(1, size - 2), random.randint(1, size - 2))

    transformed_masks: list[np.ndarray] = []
    local_masks: list[np.ndarray] = []
    for idx in range(4):
        required = {shared} if idx in (1, 3) else None
        forbidden = {shared} if idx in (0, 2) else set()
        marker_forbidden = _marker_cells(size, rotations[idx])
        success = False
        for _ in range(120):
            area = random.randint(3, 8)
            transformed = _grow_shape(size, area, required=required, forbidden=forbidden)
            if transformed is None:
                continue
            local = np.rot90(transformed, (4 - (0 if rotations[idx] is None else rotations[idx])) % 4)
            if _cells_from_mask(local) & marker_forbidden:
                continue
            transformed_masks.append(transformed)
            local_masks.append(local)
            success = True
            break
        if not success:
            return None

    overlap_count = np.zeros((size, size), dtype=int)
    for mask in transformed_masks:
        overlap_count += mask.astype(int)
    if int((overlap_count >= 2).sum()) < 1:
        return None

    expected = _overlay_bool_masks(transformed_masks, colors, STACK_ORDER)
    if len({int(v) for v in expected.ravel()} - {BG}) != 4:
        return None

    no_rotation = _overlay_bool_masks(local_masks, colors, STACK_ORDER)
    if np.array_equal(expected, no_rotation):
        return None

    wrong_order = _overlay_bool_masks(transformed_masks, colors, ALT_STACK_ORDER)
    if np.array_equal(expected, wrong_order):
        return None

    no_erase = _with_markers_then_rotate(local_masks, colors, rotations)
    if np.array_equal(expected, no_erase):
        return None

    for idx, rotation in enumerate(rotations):
        if rotation is None:
            continue
        for alt in (0, 1, 2, 3):
            if alt == rotation:
                continue
            alt_masks = list(transformed_masks)
            alt_masks[idx] = np.rot90(local_masks[idx], alt)
            alt_out = _overlay_bool_masks(alt_masks, colors, STACK_ORDER)
            if np.array_equal(expected, alt_out):
                return None

    grid = _render_input(local_masks, colors, rotations)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _sample_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid quadrant-rotation overlay sample")
