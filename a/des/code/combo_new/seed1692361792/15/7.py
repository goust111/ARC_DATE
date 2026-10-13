from __future__ import annotations

import numpy as np


BACKGROUND = 0
COLOR_POOL = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
GRID_SIZES = np.arange(12, 22, 2, dtype=int)
MAX_RETRIES = 400
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SOURCE_STEPS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
QUADRANT_STEPS = {
    "tl": (1, 1),
    "tr": (1, -1),
    "bl": (-1, 1),
    "br": (-1, -1),
}
_RNG = np.random.default_rng()


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for dr, dc in NEIGHBORS4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(np.array(coords, dtype=int))
    return components


def _l_coords(corner: tuple[int, int], dr: int, dc: int) -> np.ndarray:
    r, c = corner
    return np.array([(r, c), (r, c + dc), (r + dr, c)], dtype=int)


def _in_bounds(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.all((coords[:, 0] >= 0) & (coords[:, 0] < h) & (coords[:, 1] >= 0) & (coords[:, 1] < w))
    )


def _l_corner(coords: np.ndarray) -> tuple[int, int]:
    pts = {tuple(map(int, cell)) for cell in coords.tolist()}
    assert len(pts) == 3, "L component must contain exactly three distinct pixels"

    corner: tuple[int, int] | None = None
    degrees: list[int] = []
    for r, c in pts:
        degree = sum((r + dr, c + dc) in pts for dr, dc in NEIGHBORS4)
        degrees.append(int(degree))
        if degree == 2:
            corner = (r, c)
    assert sorted(degrees) == [1, 1, 2], "three-pixel component must be an L, not a line"
    assert corner is not None
    return corner


def _target_steps(anchor: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c = anchor
    h, w = shape
    if h % 2 == 1:
        assert r != h // 2, "anchor cannot lie on the central row"
    if w % 2 == 1:
        assert c != w // 2, "anchor cannot lie on the central column"
    dr = 1 if r < h / 2 else -1
    dc = 1 if c < w / 2 else -1
    return dr, dc


def _expanded_cells(coords: np.ndarray, h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in coords.tolist():
        r = int(r)
        c = int(c)
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                expanded.add((rr, cc))
    return expanded


def _pick_quadrant_anchor(
    h: int,
    w: int,
    quadrant: str,
    reserved_input: set[tuple[int, int]],
    reserved_output: set[tuple[int, int]],
) -> tuple[int, int] | None:
    mid_r = h // 2
    mid_c = w // 2
    dr, dc = QUADRANT_STEPS[quadrant]

    if quadrant == "tl":
        rows = range(0, mid_r - 1)
        cols = range(0, mid_c - 1)
    elif quadrant == "tr":
        rows = range(0, mid_r - 1)
        cols = range(mid_c + 1, w)
    elif quadrant == "bl":
        rows = range(mid_r + 1, h)
        cols = range(0, mid_c - 1)
    else:
        rows = range(mid_r + 1, h)
        cols = range(mid_c + 1, w)

    candidates: list[tuple[int, int]] = []
    for r in rows:
        for c in cols:
            anchor = np.array([(r, c)], dtype=int)
            out_coords = _l_coords((r, c), dr, dc)
            if any(tuple(cell) in reserved_input for cell in anchor.tolist()):
                continue
            if any(tuple(cell) in reserved_output for cell in out_coords.tolist()):
                continue
            candidates.append((int(r), int(c)))

    if not candidates:
        return None
    return candidates[int(_RNG.integers(len(candidates)))]


def _pick_source_l(
    h: int,
    w: int,
    reserved_input: set[tuple[int, int]],
) -> np.ndarray | None:
    orientations = list(SOURCE_STEPS)
    _RNG.shuffle(orientations)
    for dr, dc in orientations:
        row_range = range(0, h - 1) if dr == 1 else range(1, h)
        col_range = range(0, w - 1) if dc == 1 else range(1, w)
        candidates: list[np.ndarray] = []
        for r in row_range:
            for c in col_range:
                coords = _l_coords((int(r), int(c)), dr, dc)
                if any(tuple(cell) in reserved_input for cell in coords.tolist()):
                    continue
                candidates.append(coords)
        if candidates:
            return candidates[int(_RNG.integers(len(candidates)))]
    return None


def _render_expected(
    grid: np.ndarray,
    source_by_color: dict[int, np.ndarray],
    anchor_by_color: dict[int, tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    for color, coords in source_by_color.items():
        out[coords[:, 0], coords[:, 1]] = BACKGROUND
        dr, dc = _target_steps(anchor_by_color[color], (h, w))
        moved = _l_coords(anchor_by_color[color], dr, dc)
        out[moved[:, 0], moved[:, 1]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored three-pixel L to its same-colored anchor, rotating it so the arms point toward the grid center.

    Input structure:
    - The background is the dominant color.
    - Each foreground color appears in exactly two 4-connected pieces: one three-pixel L shape and one isolated single-pixel anchor.
    - Different colors are independent pairs, and the grid can be divided into top-left, top-right, bottom-left, and bottom-right quadrants around its center.

    Rule:
    - For each foreground color, separate its singleton anchor from its three-pixel component.
    - Verify that the three-pixel component is an L by finding the corner pixel with two 4-neighbors.
    - Clear the original L from the grid.
    - Redraw that color as a new L whose corner sits on the anchor and whose two one-pixel arms extend horizontally and vertically toward the grid center according to the anchor quadrant.

    Key insight:
    - Color identifies which anchor and source belong together, but the output rotation is controlled by the anchor's quadrant rather than by the source L's original orientation.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]

    out = g.copy()
    for color in colors:
        components = _components_4(g == color)
        sizes = sorted(len(comp) for comp in components)
        assert sizes == [1, 3], "each color must contain one singleton anchor and one three-pixel L"

        anchor_comp = next(comp for comp in components if len(comp) == 1)
        l_comp = next(comp for comp in components if len(comp) == 3)
        _l_corner(l_comp)

        out[l_comp[:, 0], l_comp[:, 1]] = bg
        anchor = tuple(map(int, anchor_comp[0].tolist()))
        dr, dc = _target_steps(anchor, (h, w))
        moved = _l_coords(anchor, dr, dc)
        assert _in_bounds(moved, h, w)
        out[moved[:, 0], moved[:, 1]] = color

    return out


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.choice(GRID_SIZES))
    w = int(_RNG.choice(GRID_SIZES))
    count = int(_RNG.integers(3, 5))
    colors = [int(x) for x in _RNG.choice(COLOR_POOL, size=count, replace=False).tolist()]

    quadrants = ["tl", "tr", "bl", "br"]
    if count == 4:
        chosen_quadrants = quadrants.copy()
    else:
        chosen_quadrants = [str(x) for x in _RNG.choice(np.array(quadrants), size=3, replace=False).tolist()]
    _RNG.shuffle(colors)
    _RNG.shuffle(chosen_quadrants)

    grid = np.full((h, w), BACKGROUND, dtype=int)
    reserved_input: set[tuple[int, int]] = set()
    reserved_output: set[tuple[int, int]] = set()
    anchor_by_color: dict[int, tuple[int, int]] = {}
    source_by_color: dict[int, np.ndarray] = {}

    for color, quadrant in zip(colors, chosen_quadrants, strict=False):
        anchor = _pick_quadrant_anchor(h, w, quadrant, reserved_input, reserved_output)
        if anchor is None:
            return None
        anchor_coords = np.array([anchor], dtype=int)
        dr, dc = QUADRANT_STEPS[quadrant]
        out_coords = _l_coords(anchor, dr, dc)

        grid[anchor[0], anchor[1]] = color
        reserved_input |= _expanded_cells(anchor_coords, h, w)
        reserved_output |= _expanded_cells(out_coords, h, w)
        anchor_by_color[color] = anchor

    color_order = colors.copy()
    _RNG.shuffle(color_order)
    for color in color_order:
        coords = _pick_source_l(h, w, reserved_input)
        if coords is None:
            return None
        grid[coords[:, 0], coords[:, 1]] = color
        reserved_input |= _expanded_cells(coords, h, w)
        source_by_color[color] = coords

    expected = _render_expected(grid, source_by_color, anchor_by_color)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid L-to-anchor transport sample")
