from __future__ import annotations

import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_PAIR_COUNT_CHOICES = (1, 2, 3)
GEN_RECT_SIDE_RANGE = (4, 9)
GEN_MAX_ATTEMPTS = 400
GEN_PAIR_SHAPE_ATTEMPTS = 80
GEN_PROFILE_ATTEMPTS = 80
GEN_PLACEMENT_ATTEMPTS = 120


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _translate(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    delta = np.array([dr, dc], dtype=int)
    return coords + delta


def _paint(out: np.ndarray, coords: np.ndarray, color: int) -> None:
    out[coords[:, 0], coords[:, 1]] = int(color)


def _find_components(grid: np.ndarray, color: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != int(color):
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(len(coords)),
                }
            )

    return comps


def _rectangle_forming_shifts(
    anchor_coords: np.ndarray,
    moving_coords: np.ndarray,
    bounds: tuple[int, int] | None,
) -> list[tuple[int, int]]:
    anchor = np.asarray(anchor_coords, dtype=int)
    moving = np.asarray(moving_coords, dtype=int)
    ar0, ac0, ar1, ac1 = _bbox(anchor)
    br0, bc0, br1, bc1 = _bbox(moving)
    total_area = int(len(anchor) + len(moving))
    anchor_set = {(int(r), int(c)) for r, c in anchor}
    matches: list[tuple[int, int]] = []

    if bounds is None:
        max_shift = total_area
        dr_range = range(-max_shift, max_shift + 1)
        dc_range = range(-max_shift, max_shift + 1)
    else:
        h, w = bounds
        dr_range = range(-br0, h - br1)
        dc_range = range(-bc0, w - bc1)

    for dr in dr_range:
        min_r = min(ar0, br0 + dr)
        max_r = max(ar1, br1 + dr)
        rect_h = max_r - min_r + 1

        for dc in dc_range:
            min_c = min(ac0, bc0 + dc)
            max_c = max(ac1, bc1 + dc)
            rect_w = max_c - min_c + 1
            if rect_h * rect_w != total_area:
                continue

            overlap = False
            for r, c in moving:
                if (int(r + dr), int(c + dc)) in anchor_set:
                    overlap = True
                    break
            if not overlap:
                matches.append((int(dr), int(dc)))

    return matches


def _solve_pair(anchor_coords: np.ndarray, moving_coords: np.ndarray, shape: tuple[int, int]) -> tuple[int, int]:
    matches = _rectangle_forming_shifts(anchor_coords, moving_coords, bounds=shape)
    assert len(matches) == 1, f"expected a unique rectangle-forming shift, got {matches}"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble each scattered same-colored fragment pair into its original solid rectangle.

    Input structure:
    - The background is a uniform color, typically black.
    - Every foreground color appears in exactly two 4-connected monochrome fragments.
    - Each same-colored pair comes from one solid axis-aligned rectangle split by a jagged cut
      that is parallel to a rectangle side.
    - The two cut boundaries are complementary, so a pure translation can interlock them.

    Rule:
    - For each foreground color, locate its two fragments and choose the fragment whose bbox is
      lexicographically earlier (smaller top row, then smaller left column) as the anchor.
    - Enumerate translations of the other fragment and keep the unique one whose union with the
      anchor forms a solid rectangle with no overlap and no holes.
    - Keep the anchor fragment fixed, move the other fragment to that matching position, and clear
      its old location to background.
    - Process each color independently on the same fixed-size canvas.

    Key insight:
    - The correct move is the only translation that turns the two irregular same-colored pieces
      into one filled rectangle, so the jagged edge acts as a geometric key-lock match.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _dominant_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != int(bg)]
    assert colors, "expected at least one foreground color"

    out = np.full_like(g, bg)
    for color in colors:
        comps = _find_components(g, color)
        assert len(comps) == 2, "each foreground color must have exactly two fragments"
        comps = sorted(comps, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
        anchor = comps[0]
        moving = comps[1]
        dr, dc = _solve_pair(anchor["coords"], moving["coords"], g.shape)
        moved_coords = _translate(moving["coords"], dr, dc)
        _paint(out, anchor["coords"], color)
        _paint(out, moved_coords, color)

    return out


def _sample_profile(length: int, span: int) -> list[int] | None:
    for _ in range(GEN_PROFILE_ATTEMPTS):
        values = [random.randint(1, span - 1)]
        for _i in range(1, length):
            prev = values[-1]
            options = [prev + d for d in (-1, 0, 1) if 1 <= prev + d <= span - 1]
            if random.random() < 0.65:
                changed = [v for v in options if v != prev]
                values.append(random.choice(changed or options))
            else:
                values.append(random.choice(options))

        changes = sum(values[i] != values[i - 1] for i in range(1, len(values)))
        if len(set(values)) >= 2 and changes >= 2:
            return values

    return None


def _piece_coords_from_profile(
    rect_h: int,
    rect_w: int,
    orientation: str,
    profile: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    first = np.zeros((rect_h, rect_w), dtype=bool)
    if orientation == "vertical":
        for r, cut in enumerate(profile):
            first[r, :cut] = True
    else:
        assert orientation == "horizontal"
        for c, cut in enumerate(profile):
            first[:cut, c] = True

    second = ~first
    return np.argwhere(first), np.argwhere(second)


def _sample_pair_shape() -> tuple[tuple[int, int], tuple[np.ndarray, np.ndarray]] | None:
    for _ in range(GEN_PAIR_SHAPE_ATTEMPTS):
        rect_h = random.randint(*GEN_RECT_SIDE_RANGE)
        rect_w = random.randint(*GEN_RECT_SIDE_RANGE)
        orientation = random.choice(("vertical", "horizontal"))
        length = rect_h if orientation == "vertical" else rect_w
        span = rect_w if orientation == "vertical" else rect_h
        profile = _sample_profile(length, span)
        if profile is None:
            continue

        piece_a, piece_b = _piece_coords_from_profile(rect_h, rect_w, orientation, profile)
        if min(len(piece_a), len(piece_b)) < 4:
            continue
        if len(_rectangle_forming_shifts(piece_a, piece_b, bounds=None)) != 1:
            continue
        return (rect_h, rect_w), (piece_a, piece_b)

    return None


def _can_place_cells(forbidden: np.ndarray, coords: np.ndarray) -> bool:
    return not bool(forbidden[coords[:, 0], coords[:, 1]].any())


def _mark_cells(forbidden: np.ndarray, coords: np.ndarray, pad: int = 1) -> None:
    h, w = forbidden.shape
    for r, c in coords:
        r0 = max(0, int(r) - pad)
        r1 = min(h, int(r) + pad + 1)
        c0 = max(0, int(c) - pad)
        c1 = min(w, int(c) + pad + 1)
        forbidden[r0:r1, c0:c1] = True


def _rect_cells(r0: int, c0: int, h: int, w: int) -> np.ndarray:
    rr, cc = np.mgrid[r0 : r0 + h, c0 : c0 + w]
    return np.stack([rr.ravel(), cc.ravel()], axis=1).astype(int)


def _bbox_top_left(coords: np.ndarray) -> tuple[int, int]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return int(r0), int(c0)


def _touches_padded_rect(coords: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = r0 - pad
    rr1 = r0 + h - 1 + pad
    cc0 = c0 - pad
    cc1 = c0 + w - 1 + pad
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(np.any((rr0 <= rows) & (rows <= rr1) & (cc0 <= cols) & (cols <= cc1)))


def _grid_shape_for_pairs(pair_count: int) -> tuple[int, int]:
    low = 12 + 4 * (pair_count - 1)
    return random.randint(low, 30), random.randint(low, 30)


def _sample_output_translation(
    grid_shape: tuple[int, int],
    rect_shape: tuple[int, int],
    anchor_local: np.ndarray,
    input_forbidden: np.ndarray,
    output_forbidden: np.ndarray,
) -> tuple[int, int] | None:
    h, w = grid_shape
    rect_h, rect_w = rect_shape

    for _ in range(GEN_PLACEMENT_ATTEMPTS):
        r0 = random.randint(0, h - rect_h)
        c0 = random.randint(0, w - rect_w)
        rect_coords = _rect_cells(r0, c0, rect_h, rect_w)
        if not _can_place_cells(output_forbidden, rect_coords):
            continue

        anchor_abs = _translate(anchor_local, r0, c0)
        if not _can_place_cells(input_forbidden, anchor_abs):
            continue
        return r0, c0

    return None


def _sample_scatter_translation(
    grid_shape: tuple[int, int],
    rect_shape: tuple[int, int],
    moving_local: np.ndarray,
    anchor_abs: np.ndarray,
    output_top_left: tuple[int, int],
    input_forbidden: np.ndarray,
) -> tuple[int, int] | None:
    h, w = grid_shape
    rect_h, rect_w = rect_shape
    anchor_key = _bbox_top_left(anchor_abs)
    out_r0, out_c0 = output_top_left

    for _ in range(GEN_PLACEMENT_ATTEMPTS):
        r0 = random.randint(0, h - rect_h)
        c0 = random.randint(0, w - rect_w)
        moving_abs = _translate(moving_local, r0, c0)
        if not _can_place_cells(input_forbidden, moving_abs):
            continue
        if _touches_padded_rect(moving_abs, out_r0, out_c0, rect_h, rect_w, pad=1):
            continue
        if _bbox_top_left(moving_abs) <= anchor_key:
            continue
        return r0, c0

    return None


def _is_solid_rectangle_component(grid: np.ndarray, color: int) -> bool:
    comps = _find_components(grid, color)
    if len(comps) != 1:
        return False
    comp = comps[0]
    r0, c0, r1, c1 = comp["bbox"]
    return int(comp["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _passes_oracle(grid: np.ndarray, expected_pairs: int) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    if out.shape != grid.shape:
        return False
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False

    colors = [int(color) for color in np.unique(grid) if int(color) != BG]
    if len(colors) != expected_pairs:
        return False

    for color in colors:
        if len(_find_components(grid, color)) != 2:
            return False
        if not _is_solid_rectangle_component(out, color):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(GEN_PAIR_COUNT_CHOICES)
    grid_shape = _grid_shape_for_pairs(pair_count)
    h, w = grid_shape
    colors = random.sample(COLORS, pair_count)
    grid = np.full((h, w), BG, dtype=int)
    input_forbidden = np.zeros((h, w), dtype=bool)
    output_forbidden = np.zeros((h, w), dtype=bool)

    for color in colors:
        sampled = _sample_pair_shape()
        if sampled is None:
            return None
        rect_shape, pieces = sampled
        rect_h, rect_w = rect_shape
        anchor_index = random.randint(0, 1)
        anchor_local = pieces[anchor_index]
        moving_local = pieces[1 - anchor_index]

        output_top_left = _sample_output_translation(grid_shape, rect_shape, anchor_local, input_forbidden, output_forbidden)
        if output_top_left is None:
            return None
        out_r0, out_c0 = output_top_left
        anchor_abs = _translate(anchor_local, out_r0, out_c0)

        scatter_top_left = _sample_scatter_translation(
            grid_shape,
            rect_shape,
            moving_local,
            anchor_abs,
            output_top_left,
            input_forbidden,
        )
        if scatter_top_left is None:
            return None
        move_r0, move_c0 = scatter_top_left
        moving_abs = _translate(moving_local, move_r0, move_c0)

        if _solve_pair(anchor_abs, moving_abs, grid_shape) != (out_r0 - move_r0, out_c0 - move_c0):
            return None

        _paint(grid, anchor_abs, color)
        _paint(grid, moving_abs, color)
        _mark_cells(input_forbidden, anchor_abs, pad=1)
        _mark_cells(input_forbidden, moving_abs, pad=1)
        _mark_cells(output_forbidden, _rect_cells(out_r0, out_c0, rect_h, rect_w), pad=1)

    if not _passes_oracle(grid, pair_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_MAX_ATTEMPTS} attempts")
