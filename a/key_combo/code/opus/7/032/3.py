from __future__ import annotations

import random
import numpy as np


BG = 0
RED = 2
YELLOW = 4
GROUND = 5
CYAN = 8
MAROON = 9
OBJECT_COLORS = {RED, CYAN}
SWAP = {RED: CYAN, CYAN: RED}

GRID_ROWS = (10, 15)
GRID_COLS = (10, 15)
HEIGHT_CHOICES = (3, 4, 5)
WIDTH_CHOICES = (2, 3)
AREA_RANGE = range(6, 11)
MAX_RETRIES = 400


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_object(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    mask = np.isin(grid, list(OBJECT_COLORS))
    comps = _find_components(mask)
    assert len(comps) == 1, "expected exactly one two-color foreground object"
    coords = comps[0]
    r0, c0, r1, c1 = _bbox(coords)
    sprite = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert c0 == 0, "expected object to touch the left wall"
    return coords, sprite, (r0, c0, r1, c1)


def _marked_axis(grid: np.ndarray) -> tuple[str, set[int]]:
    h, w = grid.shape
    row_markers = {r for r in range(h) if grid[r, 0] == YELLOW and grid[r, w - 1] == YELLOW}
    col_markers = {c for c in range(w) if grid[0, c] == YELLOW and grid[h - 1, c] == YELLOW}

    actual_markers = {(int(r), int(c)) for r, c in np.argwhere(grid == YELLOW)}
    expected_rows = {(r, 0) for r in row_markers} | {(r, w - 1) for r in row_markers}
    expected_cols = {(0, c) for c in col_markers} | {(h - 1, c) for c in col_markers}

    row_ok = bool(row_markers) and actual_markers == expected_rows
    col_ok = bool(col_markers) and actual_markers == expected_cols
    assert row_ok != col_ok, "expected opposite-edge marker pairs on exactly one axis"
    if row_ok:
        return "row", row_markers
    return "col", col_markers


def _sheared_pixels(sprite: np.ndarray, base_col: int) -> list[tuple[int, int, int]]:
    h, w = sprite.shape
    pixels: list[tuple[int, int, int]] = []
    for r in range(h):
        shear = max(0, r - 2)
        for c in range(w):
            color = int(sprite[r, c])
            if color not in OBJECT_COLORS:
                continue
            pixels.append((r, base_col + c - shear, color))
    return pixels


def _transform_object(
    grid: np.ndarray,
    marker_axis: str,
    marker_indices: set[int],
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    coords, sprite, bbox = _extract_object(grid)
    r0, _c0, r1, c1 = bbox
    out = grid.copy()
    out[coords[:, 0], coords[:, 1]] = BG

    mirrored = np.fliplr(sprite)
    base_col = grid.shape[1] - mirrored.shape[1]
    placed = _sheared_pixels(mirrored, base_col)

    for local_r, col, color in placed:
        row = r0 + local_r
        on_marker = row in marker_indices if marker_axis == "row" else col in marker_indices
        new_color = SWAP[color] if on_marker else color
        out[row, col] = new_color

    out[-1, :] = MAROON
    return out, placed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror a left-wall two-color object to the right wall with row-wise shear, then swap colors only on marked rows.

    Input structure:
    - The grid has a black background and a full gray ground row on the bottom edge.
    - One 4-connected foreground object made of red and cyan pixels touches the left wall and sits directly above the ground.
    - Yellow marker pixels appear as opposite-edge pairs selecting one or two special horizontal or vertical guide lines.

    Rule:
    - Erase the original object.
    - Horizontally mirror it, place the mirror against the right wall with the same vertical span, and shear rows so the top three rows stay flush right while each lower row shifts one extra cell left.
    - For every placed object pixel lying on a marker-selected guide line, swap red and cyan; keep all other placed pixels unchanged.
    - Recolor the ground row from the original wall-side start through the new right-wall destination to maroon, keeping yellow markers unchanged.

    Key insight:
    - The shear changes which final object pixels land on the selected guide lines, so color swapping must be decided after relocation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 4 and w >= 4, "grid too small"
    assert np.all(np.isin(g[-1, :], [GROUND, YELLOW])), "expected a gray ground row with optional yellow markers"

    marker_axis, marker_indices = _marked_axis(g)
    assert marker_indices, "expected at least one marker pair"
    if marker_axis == "row":
        assert all(0 < r < h - 1 for r in marker_indices), "marked rows must be interior"
    else:
        assert all(0 < c < w - 1 for c in marker_indices), "marked columns must be interior"

    obj_coords, sprite, bbox = _extract_object(g)
    r0, c0, r1, c1 = bbox
    assert c0 == 0
    assert r1 == h - 2, "object must sit directly on top of the ground"
    assert 3 <= sprite.shape[0] <= 5 and 2 <= sprite.shape[1] <= 3
    assert sprite.shape[0] + sprite.shape[1] <= w, "object too wide for relocation"

    out, placed = _transform_object(g, marker_axis, marker_indices)
    for _local_r, col, _color in placed:
        assert 0 <= col < w, "sheared object must stay in bounds"

    out[g == YELLOW] = YELLOW
    return out


def _row_patterns(height: int, width: int) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    full = np.ones((height, width), dtype=bool)
    candidates.append(full)

    if width >= 3:
        candidates.append(np.array([[1, 1, 0]] * height, dtype=bool)[:, :width])
    if height >= 4:
        patt = np.ones((height, width), dtype=bool)
        patt[1, min(width - 1, 1)] = False
        candidates.append(patt)
    if height >= 5:
        patt = np.ones((height, width), dtype=bool)
        patt[2, width - 1] = False
        candidates.append(patt)

    unique: list[np.ndarray] = []
    seen = set()
    for patt in candidates:
        area = int(patt.sum())
        if area not in AREA_RANGE:
            continue
        if len(_find_components(patt)) != 1:
            continue
        key = tuple(map(int, patt.ravel()))
        if key not in seen:
            seen.add(key)
            unique.append(patt)
    return unique


def _make_object_sprite() -> np.ndarray:
    height = random.choice(HEIGHT_CHOICES)
    width = random.choice(WIDTH_CHOICES)
    patterns = _row_patterns(height, width)
    assert patterns, "expected at least one connected object template"
    mask = random.choice(patterns)
    assert mask[:, 0].any(), "object must touch the left wall"
    assert mask[-1].any(), "object must rest on the ground"

    colors = np.full(mask.shape, BG, dtype=int)
    cells = np.argwhere(mask)
    random.shuffle(cells)
    cyan_count = random.randint(1, len(cells) - 1)
    cyan_cells = {tuple(map(int, rc)) for rc in cells[:cyan_count]}
    for r, c in np.argwhere(mask):
        colors[r, c] = CYAN if (int(r), int(c)) in cyan_cells else RED
    return colors


def _candidate_marker_cols(sprite: np.ndarray, grid_cols: int) -> list[int]:
    base_col = grid_cols - sprite.shape[1]
    hits_by_col: dict[int, int] = {}
    placed = _sheared_pixels(np.fliplr(sprite), base_col)
    final_cols = [col for _r, col, _color in placed]
    for col in sorted(set(final_cols)):
        hits_by_col[col] = sum(1 for c in final_cols if c == col)

    cols = []
    for col, count in hits_by_col.items():
        if 0 < count < len(placed) and 0 < col < grid_cols - 1:
            cols.append(col)
    return cols


def _build_input() -> np.ndarray | None:
    rows = random.randint(*GRID_ROWS)
    cols = random.randint(*GRID_COLS)
    grid = np.zeros((rows, cols), dtype=int)
    grid[-1, :] = GROUND

    sprite = _make_object_sprite()
    sh, sw = sprite.shape
    obj_top = rows - 1 - sh
    grid[obj_top : rows - 1, 0:sw] = sprite

    candidates = _candidate_marker_cols(sprite, cols)
    if not candidates:
        return None

    marker_count = 1 if len(candidates) == 1 else random.choice((1, 2))
    chosen_cols = sorted(random.sample(candidates, marker_count))
    for c in chosen_cols:
        grid[0, c] = YELLOW
        grid[rows - 1, c] = YELLOW

    out = main(grid)

    # Reconstruct the sheared object to validate the key branch coverage.
    marker_cols = set(chosen_cols)
    _, placed = _transform_object(grid, "col", marker_cols)
    on_mark = sum(1 for _local_r, col, _color in placed if col in marker_cols)
    off_mark = len(placed) - on_mark
    if on_mark == 0 or off_mark == 0:
        return None

    # The moved object should be the only non-marker foreground above the ground.
    fg_mask = (out[:-1] != BG) & (out[:-1] != YELLOW)
    comps = _find_components(fg_mask)
    if len(comps) != 1:
        return None

    bottom_ok = np.all((out[-1, :] == MAROON) | (out[-1, :] == YELLOW))
    if not bottom_ok:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
