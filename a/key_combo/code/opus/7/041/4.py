from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
OBJ = 1
RED = 2
GREEN = 3
YELLOW = 4
MARKER = 7

MISSING_TO_SEQUENCE = {
    (0, 0): (GREEN, YELLOW, RED),
    (0, 1): (RED, GREEN, YELLOW),
    (1, 0): (YELLOW, RED, GREEN),
    (1, 1): (GREEN, RED, YELLOW),
}
MISSING_CORNERS = tuple(MISSING_TO_SEQUENCE)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_ATTEMPTS = 400


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _marker_missing_corner(grid: np.ndarray, coords: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    assert r1 - r0 + 1 == 2 and c1 - c0 + 1 == 2 and coords.shape[0] == 3
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    missing = np.argwhere(patch == BG)
    assert missing.shape == (1, 2)
    return tuple(map(int, missing[0]))


def _full_support(mask: np.ndarray) -> np.ndarray:
    supported = np.zeros_like(mask, dtype=bool)
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return supported
    supported[1:-1, 1:-1] = (
        mask[:-2, :-2]
        & mask[:-2, 1:-1]
        & mask[:-2, 2:]
        & mask[1:-1, :-2]
        & mask[1:-1, 1:-1]
        & mask[1:-1, 2:]
        & mask[2:, :-2]
        & mask[2:, 1:-1]
        & mask[2:, 2:]
    )
    return supported


def _erosion_levels(component_mask: np.ndarray) -> list[np.ndarray]:
    levels: list[np.ndarray] = []
    current = component_mask.copy()
    while True:
        supported = _full_support(current)
        if not supported.any():
            break
        levels.append(np.argwhere(supported))
        current = supported
    return levels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase each orange L marker and recolor each thick blue component by a marker-chosen cyclic sequence over nested erosion layers.

    Input structure:
    - The background is black, with 3-4 disjoint blue 4-connected components made from thick solid blocks plus one-cell-wide corridors.
    - Each blue component has exactly one orange 2x2 L-triomino marker directly above its bounding box, with overlapping columns.
    - The missing corner of the marker encodes an ordered 3-color sequence rather than a single color.

    Rule:
    - Remove all orange marker pixels to black.
    - For each blue component, read the missing marker corner and map it to one cyclic sequence:
      top-left -> [green, yellow, red], top-right -> [red, green, yellow],
      bottom-left -> [yellow, red, green], bottom-right -> [green, red, yellow].
    - Repeatedly keep only those component pixels whose full 3x3 neighborhood stays inside the current set; color the k-th surviving nested set with sequence[k mod 3], leaving uncovered border and corridor pixels blue.

    Key insight:
    The L marker encodes the order of colors, while repeated 3x3 erosion creates nested interior layers whose final shells appear because deeper layers overwrite smaller subsets of earlier ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    markers = _components_of_color(g, MARKER)
    objects = _components_of_color(g, OBJ)
    assert 3 <= len(objects) <= 4
    assert len(markers) == len(objects)

    marker_info = []
    for coords in markers:
        missing = _marker_missing_corner(g, coords)
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        marker_info.append((r0, c0, r1, c1, missing))

    out = g.copy()
    out[out == MARKER] = BG

    used = set()
    for obj_coords in objects:
        r0, c0, _r1, c1 = _bbox_from_coords(obj_coords)
        candidates = []
        for idx, (mr0, mc0, mr1, mc1, missing) in enumerate(marker_info):
            if mr1 + 1 == r0 and max(c0, mc0) <= min(c1, mc1):
                candidates.append((idx, missing))
        assert len(candidates) == 1
        idx, missing = candidates[0]
        used.add(idx)

        comp_mask = np.zeros(g.shape, dtype=bool)
        comp_mask[obj_coords[:, 0], obj_coords[:, 1]] = True
        sequence = MISSING_TO_SEQUENCE[missing]
        for depth, level_coords in enumerate(_erosion_levels(comp_mask)):
            color = sequence[depth % 3]
            out[level_coords[:, 0], level_coords[:, 1]] = color

    assert len(used) == len(markers)
    return out


def _rect_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _build_component_mask(rng: random.Random) -> np.ndarray:
    main_h = rng.randint(7, 8)
    main_w = rng.randint(7, 8)
    aux_h = rng.randint(3, 5)
    aux_w = rng.randint(3, 5)
    corridor_len = rng.randint(1, 2)
    direction = rng.choice(("left", "right", "up", "down"))

    cells = _rect_cells(0, 0, main_h, main_w)

    if direction in ("left", "right"):
        main_anchor = rng.randint(2, main_h - 3)
        aux_anchor = rng.randint(1, aux_h - 2)
        aux_top = main_anchor - aux_anchor
        if direction == "right":
            aux_left = main_w + corridor_len
            for dc in range(corridor_len):
                cells.add((main_anchor, main_w + dc))
        else:
            aux_left = -(corridor_len + aux_w)
            for dc in range(corridor_len):
                cells.add((main_anchor, -1 - dc))
        cells |= _rect_cells(aux_top, aux_left, aux_h, aux_w)
    else:
        main_anchor = rng.randint(2, main_w - 3)
        aux_anchor = rng.randint(1, aux_w - 2)
        aux_left = main_anchor - aux_anchor
        if direction == "down":
            aux_top = main_h + corridor_len
            for dr in range(corridor_len):
                cells.add((main_h + dr, main_anchor))
        else:
            aux_top = -(corridor_len + aux_h)
            for dr in range(corridor_len):
                cells.add((-1 - dr, main_anchor))
        cells |= _rect_cells(aux_top, aux_left, aux_h, aux_w)

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    min_r, min_c = min(rows), min(cols)
    shifted = {(r - min_r, c - min_c) for r, c in cells}
    max_r = max(r for r, _ in shifted)
    max_c = max(c for _, c in shifted)
    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    for r, c in shifted:
        mask[r, c] = True
    return mask


def _make_group_patch(rng: random.Random, missing: tuple[int, int]) -> np.ndarray:
    comp_mask = _build_component_mask(rng)
    levels = _erosion_levels(comp_mask)
    assert len(levels) >= 2

    h, w = comp_mask.shape
    patch = np.zeros((h + 2, w), dtype=int)
    patch[2:][comp_mask] = OBJ

    marker_left = rng.randint(0, w - 2)
    patch[0 : 2, marker_left : marker_left + 2] = MARKER
    patch[missing[0], marker_left + missing[1]] = BG
    return patch


def _layout_patches(rng: random.Random, patches: list[np.ndarray]) -> np.ndarray | None:
    gap = rng.randint(2, 3)
    slots = rng.sample([(0, 0), (0, 1), (1, 0), (1, 1)], k=len(patches))

    row_heights = [0, 0]
    col_widths = [0, 0]
    for patch, (row, col) in zip(patches, slots):
        row_heights[row] = max(row_heights[row], patch.shape[0])
        col_widths[col] = max(col_widths[col], patch.shape[1])

    used_rows = [idx for idx, size in enumerate(row_heights) if size > 0]
    used_cols = [idx for idx, size in enumerate(col_widths) if size > 0]
    total_h = sum(row_heights[idx] for idx in used_rows) + gap * (len(used_rows) - 1)
    total_w = sum(col_widths[idx] for idx in used_cols) + gap * (len(used_cols) - 1)
    if total_h > 30 or total_w > 30:
        return None

    extra_h = min(30 - total_h, 4)
    extra_w = min(30 - total_w, 4)
    top_margin = rng.randint(0, extra_h)
    left_margin = rng.randint(0, extra_w)
    grid = np.zeros((total_h + top_margin, total_w + left_margin), dtype=int)

    row_offsets: dict[int, int] = {}
    cursor = top_margin
    for pos, row in enumerate(used_rows):
        if pos > 0:
            cursor += gap
        row_offsets[row] = cursor
        cursor += row_heights[row]

    col_offsets: dict[int, int] = {}
    cursor = left_margin
    for pos, col in enumerate(used_cols):
        if pos > 0:
            cursor += gap
        col_offsets[col] = cursor
        cursor += col_widths[col]

    for patch, (row, col) in zip(patches, slots):
        r_space = row_heights[row] - patch.shape[0]
        c_space = col_widths[col] - patch.shape[1]
        r0 = row_offsets[row] + rng.randint(0, r_space)
        c0 = col_offsets[col] + rng.randint(0, c_space)
        view = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        if np.any((view != BG) & (patch != BG)):
            return None
        view[patch != BG] = patch[patch != BG]

    return grid


def _validate_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not set(map(int, np.unique(grid))) <= {BG, OBJ, MARKER}:
        return False

    objects = _components_of_color(grid, OBJ)
    markers = _components_of_color(grid, MARKER)
    if len(objects) != len(markers) or not (3 <= len(objects) <= 4):
        return False

    missing_corners = []
    for coords in markers:
        try:
            missing_corners.append(_marker_missing_corner(grid, coords))
        except AssertionError:
            return False
    if len(set(missing_corners)) < 2:
        return False

    for obj_coords in objects:
        r0, c0, _r1, c1 = _bbox_from_coords(obj_coords)
        matches = 0
        comp_mask = np.zeros(grid.shape, dtype=bool)
        comp_mask[obj_coords[:, 0], obj_coords[:, 1]] = True
        if len(_erosion_levels(comp_mask)) < 2:
            return False
        for marker_coords in markers:
            mr0, mc0, mr1, mc1 = _bbox_from_coords(marker_coords)
            if mr1 + 1 == r0 and max(c0, mc0) <= min(c1, mc1):
                matches += 1
        if matches != 1:
            return False

    try:
        out = main(grid)
    except Exception:
        return False
    if out.shape != grid.shape:
        return False
    return not np.array_equal(out, grid)


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    count = rng.choice((3, 4))
    chosen_missing = rng.sample(MISSING_CORNERS, k=count)
    patches = [_make_group_patch(rng, missing) for missing in chosen_missing]
    grid = _layout_patches(rng, patches)
    if grid is None or not _validate_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        rng = random.Random()
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
