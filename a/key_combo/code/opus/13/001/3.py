from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
GREEN = 3
YELLOW = 4
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    cells = {tuple(map(int, rc)) for rc in coords}
    stack = [next(iter(cells))]
    seen = set()
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        r, c = cell
        for dr, dc in DIR4:
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                stack.append(nb)
    return len(seen) == len(cells)


def _valid_vertical_axes(mask: np.ndarray) -> list[int]:
    h, w = mask.shape
    cells = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    valid = []
    for axis2 in range(2 * w - 1):
        reflected = set()
        ok = True
        for r, c in cells:
            nc = axis2 - c
            if not (0 <= nc < w):
                ok = False
                break
            reflected.add((r, nc))
        if ok and reflected == cells:
            valid.append(axis2)
    return valid


def _valid_horizontal_axes(mask: np.ndarray) -> list[int]:
    h, w = mask.shape
    cells = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    valid = []
    for axis2 in range(2 * h - 1):
        reflected = set()
        ok = True
        for r, c in cells:
            nr = axis2 - r
            if not (0 <= nr < h):
                ok = False
                break
            reflected.add((nr, c))
        if ok and reflected == cells:
            valid.append(axis2)
    return valid


def _detect_unique_axis(mask: np.ndarray) -> tuple[str, int]:
    vertical = _valid_vertical_axes(mask)
    horizontal = _valid_horizontal_axes(mask)
    axes = [("vertical", axis2) for axis2 in vertical] + [
        ("horizontal", axis2) for axis2 in horizontal
    ]
    assert len(axes) == 1, "expected exactly one horizontal or vertical mirror axis"
    return axes[0]


def _slide_mask(mask: np.ndarray, direction: str) -> np.ndarray:
    h, w = mask.shape
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    dr = dc = 0
    if direction == "left":
        dc = -c0
    elif direction == "right":
        dc = (w - 1) - c1
    elif direction == "up":
        dr = -r0
    else:
        assert direction == "down"
        dr = (h - 1) - r1

    out = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        out[int(r) + dr, int(c) + dc] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the unique mirror-symmetric blue shape, decode direction from anomaly colors, and slide it to the chosen edge.

    Input structure:
    - A black background contains one central 4-connected foreground shape.
    - Most foreground pixels are blue; 1-2 foreground pixels are green or yellow anomalies.
    - Recoloring those anomalies back to blue yields a shape with exactly one mirror axis, either vertical or horizontal.

    Rule:
    - Treat every non-black cell as part of the same shape and record the non-blue anomaly colors.
    - Temporarily restore all anomalies to blue, detect the unique mirror-axis orientation, and decode green/yellow in that axis-dependent coordinate system.
    - Green means move in the positive axis direction and yellow means move in the negative one; with a vertical axis this is right/left, and with a horizontal axis this is down/up.
    - Use majority vote over anomaly colors, restore the whole shape to blue, erase the original position, and translate the shape until its bounding box is flush with the chosen grid edge.

    Key insight:
    - The anomaly colors do not encode absolute directions; they only become meaningful after detecting whether the recovered shape is vertically or horizontally symmetric.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "grid exceeds ARC size limits"

    shape_mask = g != BG
    assert shape_mask.any(), "expected a foreground shape"
    assert _is_connected(shape_mask), "foreground must be 4-connected after restoring anomalies"

    anomaly_mask = (g != BG) & (g != BLUE)
    anomaly_colors = [int(v) for v in g[anomaly_mask]]
    assert anomaly_colors, "expected at least one anomaly pixel"
    assert set(anomaly_colors).issubset({GREEN, YELLOW}), "unexpected anomaly color"

    green_count = anomaly_colors.count(GREEN)
    yellow_count = anomaly_colors.count(YELLOW)
    assert green_count != yellow_count, "direction vote must not tie"

    axis_kind, _axis2 = _detect_unique_axis(shape_mask)
    if axis_kind == "vertical":
        direction = "right" if green_count > yellow_count else "left"
    else:
        direction = "down" if green_count > yellow_count else "up"

    moved_mask = _slide_mask(shape_mask, direction)
    out = np.zeros_like(g)
    out[moved_mask] = BLUE
    return out


def _orbit_cells(r: int, c: int, w: int) -> tuple[tuple[int, int], ...]:
    mirror_c = w - 1 - c
    if mirror_c == c:
        return ((r, c),)
    if c < mirror_c:
        return ((r, c), (r, mirror_c))
    return ((r, mirror_c), (r, c))


def _add_orbit(mask: np.ndarray, orbit: tuple[tuple[int, int], ...]) -> None:
    for r, c in orbit:
        mask[r, c] = True


def _orbit_adjacent(mask: np.ndarray, orbit: tuple[tuple[int, int], ...]) -> bool:
    for r, c in orbit:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                return True
    return False


def _shape_side_counts(mask: np.ndarray, axis_kind: str, axis2: int) -> tuple[int, int]:
    neg = pos = 0
    for r, c in np.argwhere(mask):
        if axis_kind == "vertical":
            delta = 2 * int(c) - axis2
        else:
            delta = 2 * int(r) - axis2
        if delta < 0:
            neg += 1
        elif delta > 0:
            pos += 1
    return neg, pos


def _cells_on_side(mask: np.ndarray, axis_kind: str, axis2: int, sign: int) -> list[tuple[int, int]]:
    cells = []
    for r, c in np.argwhere(mask):
        delta = 2 * int(c) - axis2 if axis_kind == "vertical" else 2 * int(r) - axis2
        if delta * sign > 0:
            cells.append((int(r), int(c)))
    return cells


def _build_vertical_shape(height: int, width: int, target_area: int) -> np.ndarray | None:
    half_cols = (width + 1) // 2
    all_orbits = [_orbit_cells(r, c, width) for r in range(height) for c in range(half_cols)]
    seam_col = width // 2 if width % 2 == 1 else (width // 2) - 1
    seed_orbits = [_orbit_cells(r, seam_col, width) for r in range(height)]

    for _ in range(SHAPE_ATTEMPTS):
        mask = np.zeros((height, width), dtype=bool)
        _add_orbit(mask, random.choice(seed_orbits))

        while int(mask.sum()) < target_area:
            current_area = int(mask.sum())
            candidates = []
            for orbit in all_orbits:
                if any(mask[r, c] for r, c in orbit):
                    continue
                if current_area + len(orbit) > target_area:
                    continue
                if _orbit_adjacent(mask, orbit):
                    candidates.append(orbit)
            if not candidates:
                break
            _add_orbit(mask, random.choice(candidates))

        if int(mask.sum()) != target_area or not _is_connected(mask):
            continue

        try:
            axis_kind, axis2 = _detect_unique_axis(mask)
        except AssertionError:
            continue
        if axis_kind != "vertical":
            continue
        neg, pos = _shape_side_counts(mask, axis_kind, axis2)
        if min(neg, pos) == 0:
            continue
        return mask
    return None


def _make_symmetric_shape(axis_kind: str) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        height = random.randint(4, 7)
        width = random.randint(3, 7)
        max_area = min(12, height * width)
        if max_area < 6:
            continue
        target_area = random.randint(6, max_area)
        mask = _build_vertical_shape(height, width, target_area)
        if mask is None:
            continue
        if axis_kind == "horizontal":
            mask = mask.T.copy()
            try:
                found_kind, axis2 = _detect_unique_axis(mask)
            except AssertionError:
                continue
            if found_kind != "horizontal":
                continue
        else:
            found_kind, axis2 = _detect_unique_axis(mask)
        counts = _shape_side_counts(mask, found_kind, axis2)
        if max(counts) < 2:
            continue
        return mask
    return None


def _place_shape(shape_mask: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray | None:
    h, w = grid_shape
    sh, sw = shape_mask.shape
    row_lo, row_hi = 2, h - sh - 2
    col_lo, col_hi = 2, w - sw - 2
    if row_lo > row_hi or col_lo > col_hi:
        return None
    r0 = random.randint(row_lo, row_hi)
    c0 = random.randint(col_lo, col_hi)
    grid = np.zeros((h, w), dtype=int)
    grid[r0 : r0 + sh, c0 : c0 + sw][shape_mask] = BLUE
    return grid


def _inject_anomalies(grid: np.ndarray) -> np.ndarray | None:
    shape_mask = grid == BLUE
    axis_kind, axis2 = _detect_unique_axis(shape_mask)
    sign = random.choice([-1, 1])
    candidates = _cells_on_side(shape_mask, axis_kind, axis2, sign)
    if not candidates:
        return None

    anomaly_count = 1 if len(candidates) == 1 or random.random() < 0.5 else 2
    if anomaly_count == 2 and len(candidates) < 2:
        anomaly_count = 1

    chosen = random.sample(candidates, anomaly_count)
    anomaly_color = random.choice([GREEN, YELLOW])
    out = grid.copy()
    for r, c in chosen:
        out[r, c] = anomaly_color

    return out


def _generate_attempt() -> np.ndarray | None:
    grid_h = random.randint(11, 15)
    grid_w = random.randint(11, 15)
    axis_kind = random.choice(["vertical", "horizontal"])
    restored_shape = _make_symmetric_shape(axis_kind)
    if restored_shape is None:
        return None

    grid = _place_shape(restored_shape, (grid_h, grid_w))
    if grid is None:
        return None
    inp = _inject_anomalies(grid)
    if inp is None:
        return None

    try:
        out = main(inp)
    except AssertionError:
        return None

    if out.shape != inp.shape or not np.all((out == BG) | (out == BLUE)):
        return None
    if np.array_equal(out, inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
