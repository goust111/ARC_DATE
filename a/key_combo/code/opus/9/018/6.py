from __future__ import annotations

import numpy as np


BG = 0
INPUT_CROSS_COLORS = (7, 8)
INPUT_MASK_COLORS = (2, 3, 6)
OUTPUT_CYCLE = (1, 2, 3, 4)

GRID_SIZE_CHOICES = (11, 13)
MASK_AREA_RANGE = (4, 8)
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    """arc_utils.components_mask-style 4-connected components as (N,2) coord arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _find_cross(grid: np.ndarray) -> tuple[int, int]:
    n, m = map(int, grid.shape)
    assert n == m and n % 2 == 1, "expected an odd square grid"
    full_rows = [(r, int(grid[r, 0])) for r in range(n) if int(grid[r, 0]) != BG and np.all(grid[r, :] == grid[r, 0])]
    full_cols = [(c, int(grid[0, c])) for c in range(m) if int(grid[0, c]) != BG and np.all(grid[:, c] == grid[0, c])]
    matches = [(r, col) for r, col in full_rows for c, col2 in full_cols if r == c and col == col2]
    assert len(matches) == 1, "expected one full horizontal/vertical cross"
    center, cross_color = matches[0]
    assert center == n // 2, "cross must pass through the grid center"
    return int(cross_color), int(center)


def _mirrored_coords(coords: np.ndarray, center: int) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        pts.add((rr, cc))
        pts.add((rr, 2 * center - cc))
        pts.add((2 * center - rr, cc))
        pts.add((2 * center - rr, 2 * center - cc))
    return pts


def _extract_ul_component(grid: np.ndarray, cross_color: int, center: int) -> np.ndarray:
    ul = (grid[:center, :center] != BG) & (grid[:center, :center] != int(cross_color))
    comps = _components_mask(ul)
    assert len(comps) == 1, "expected exactly one upper-left foreground component"
    return comps[0]


def _expected_from_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    cross_color, center = _find_cross(g)
    ul_coords = _extract_ul_component(g, cross_color, center)
    all_pts = _mirrored_coords(ul_coords, center)
    columns = sorted({c for _, c in all_pts})
    col_to_color = {c: int(OUTPUT_CYCLE[i % len(OUTPUT_CYCLE)]) for i, c in enumerate(columns)}

    out = np.zeros_like(g)
    for r, c in all_pts:
        out[r, c] = col_to_color[c]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the center cross, mirror the upper-left mask into all quadrants, then recolor by mirrored column order.

    Input structure:
    - The grid is an odd square on a black background with a unique center.
    - One full horizontal line and one full vertical line of the same non-black color form a center cross.
    - Outside the cross, exactly one 4-connected single-color mask appears in the upper-left quadrant, and the other quadrants are empty.

    Rule:
    - Remove the full cross to background.
    - Treat the upper-left mask as the template and reflect it across the center column, center row, and both axes to populate all four quadrants.
    - Collect the distinct occupied columns of the mirrored result from left to right and color them cyclically as blue, red, green, yellow, repeating every four columns.
    - Paint every foreground pixel with the color assigned to its column and leave all other cells black.

    Key insight:
    - The cross is both the thing to erase and the unique reference frame for the reflections; only after the four-quadrant mirror is built can the final left-to-right column recoloring be determined.
    """
    return _expected_from_rule(np.asarray(grid, dtype=int))


def _cropped_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _shape_ok(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if not (MASK_AREA_RANGE[0] <= int(coords.shape[0]) <= MASK_AREA_RANGE[1]):
        return False
    if len(np.unique(coords[:, 1])) < 3:
        return False
    if len(np.unique(coords[:, 0])) < 2:
        return False
    cropped = _cropped_mask(mask)
    if np.array_equal(cropped, np.fliplr(cropped)):
        return False
    if np.array_equal(cropped, np.flipud(cropped)):
        return False
    if np.array_equal(cropped, np.rot90(cropped, 2)):
        return False
    return True


def _grow_shape(side: int, area: int, rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        mask = np.zeros((side, side), dtype=bool)
        sr = int(rng.integers(0, side))
        sc = int(rng.integers(0, side))
        mask[sr, sc] = True
        frontier = {cell for cell in _neighbors4(sr, sc) if 0 <= cell[0] < side and 0 <= cell[1] < side}
        while int(mask.sum()) < area and frontier:
            idx = int(rng.integers(0, len(frontier)))
            cell = tuple(frontier)[idx]
            frontier.remove(cell)
            r, c = cell
            if mask[r, c]:
                continue
            mask[r, c] = True
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < side and 0 <= nc < side and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if int(mask.sum()) == area and _shape_ok(mask):
            return mask
    return None


def _validate_input(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    cross_color, center = _find_cross(g)
    assert cross_color in INPUT_CROSS_COLORS

    non_bg = set(map(int, np.unique(g))) - {BG}
    assert cross_color in non_bg
    other_colors = non_bg - {cross_color}
    assert len(other_colors) == 1
    mask_color = next(iter(other_colors))
    assert mask_color in INPUT_MASK_COLORS and mask_color != cross_color

    ul_coords = _extract_ul_component(g, cross_color, center)
    assert MASK_AREA_RANGE[0] <= int(ul_coords.shape[0]) <= MASK_AREA_RANGE[1]
    assert len(np.unique(ul_coords[:, 1])) >= 3

    assert np.all(g[center, :] == cross_color)
    assert np.all(g[:, center] == cross_color)
    assert not np.any(g[:center, center + 1 :] != BG)
    assert not np.any(g[center + 1 :, :center] != BG)
    assert not np.any(g[center + 1 :, center + 1 :] != BG)
    return cross_color, center, ul_coords


def _validate_output(inp: np.ndarray, out: np.ndarray) -> None:
    _, center, ul_coords = _validate_input(inp)
    expected_pts = _mirrored_coords(ul_coords, center)
    out_pts = {tuple(map(int, rc)) for rc in np.argwhere(out != BG)}
    assert out_pts == expected_pts, "output foreground locations must be exactly the mirrored mask"
    assert np.all(out[center, :] == BG)
    assert np.all(out[:, center] == BG)

    cols = sorted({c for _, c in expected_pts})
    for idx, col in enumerate(cols):
        want = int(OUTPUT_CYCLE[idx % len(OUTPUT_CYCLE)])
        rows = [r for r, c in expected_pts if c == col]
        assert rows, "occupied column cannot be empty"
        assert set(map(int, out[rows, col])) == {want}

    allowed = {BG, *OUTPUT_CYCLE}
    assert set(map(int, np.unique(out))).issubset(allowed)


def _build_input(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.choice(GRID_SIZE_CHOICES))
    center = n // 2
    area = int(rng.integers(MASK_AREA_RANGE[0], MASK_AREA_RANGE[1] + 1))
    mask = _grow_shape(center, area, rng)
    if mask is None:
        return None

    cross_color = int(rng.choice(INPUT_CROSS_COLORS))
    mask_color = int(rng.choice([c for c in INPUT_MASK_COLORS if c != cross_color]))

    grid = np.zeros((n, n), dtype=int)
    grid[center, :] = cross_color
    grid[:, center] = cross_color
    grid[:center, :center][mask] = mask_color
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _build_input(rng)
        if grid is None:
            continue
        out = _expected_from_rule(grid)
        try:
            _validate_output(grid, out)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid mirrored-column ARC instance")
