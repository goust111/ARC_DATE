from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
TEAL = 8

MARKER_TO_OUTLINE = {
    GREEN: PINK,
    RED: BLUE,
    TEAL: YELLOW,
}

MARKER_COLORS = tuple(MARKER_TO_OUTLINE)
CARDINAL_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

SYMMETRIC_MARKERS = (
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=int,
    ),
    np.ones((3, 3), dtype=int),
    np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=int,
    ),
)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in CARDINAL_DIRS:
        yield r + dr, c + dc


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    offsets = CARDINAL_DIRS if connectivity == 4 else tuple(
        (dr, dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    )
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in offsets:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_coords(coords: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0 = min(rows), min(cols)
    shifted = {(r - r0, c - c0) for r, c in coords}
    return shifted, max(r for r, _ in shifted) + 1, max(c for _, c in shifted) + 1


def _coords_to_mask(coords: set[tuple[int, int]]) -> np.ndarray:
    norm, h, w = _normalize_coords(coords)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _is_hv_symmetric(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.flipud(mask)) and np.array_equal(mask, np.fliplr(mask)))


def _find_axis_object(grid: np.ndarray) -> tuple[int, int, dict[tuple[int, int], int], tuple[int, int, int, int]]:
    blue_components = _components_mask(grid == BLUE, connectivity=4)
    candidates = []
    for comp in blue_components:
        r0, c0, r1, c1 = _bbox_coords(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h % 2 == 0 or w % 2 == 0:
            continue
        mask = np.zeros((h, w), dtype=bool)
        mask[comp[:, 0] - r0, comp[:, 1] - c0] = True
        if _is_hv_symmetric(mask):
            candidates.append((comp, (r0, c0, r1, c1)))

    assert len(candidates) == 1, "expected exactly one blue object with both mirror axes"
    comp, bbox = candidates[0]
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    pixels = {(int(r), int(c)): BLUE for r, c in comp}
    return center_r, center_c, pixels, bbox


def _extract_sprite_infos(grid: np.ndarray) -> list[dict[str, object]]:
    grey_components = _components_mask(grid == GREY, connectivity=4)
    assert 1 <= len(grey_components) <= 2, "expected one or two grey sprites"

    component_of: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(grey_components):
        for r, c in comp:
            component_of[(int(r), int(c))] = idx

    markers = [
        (int(r), int(c), int(grid[r, c]))
        for r, c in np.argwhere(np.isin(grid, MARKER_COLORS))
    ]
    assert len(markers) == len(grey_components), "expected one colored marker per sprite"

    markers_by_component: dict[int, tuple[int, int, int]] = {}
    for r, c, color in markers:
        touched = set()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and (nr, nc) in component_of:
                touched.add(component_of[(nr, nc)])
        assert len(touched) == 1, "each marker must touch exactly one grey sprite"
        idx = next(iter(touched))
        assert idx not in markers_by_component, "each grey sprite must have a unique marker"
        markers_by_component[idx] = (r, c, color)

    assert len(markers_by_component) == len(grey_components), "every sprite needs a marker"

    infos: list[dict[str, object]] = []
    for idx, comp in enumerate(grey_components):
        marker_r, marker_c, marker_color = markers_by_component[idx]
        body = {(int(r), int(c)) for r, c in comp}
        outline_color = MARKER_TO_OUTLINE[marker_color]
        outline = set()
        for r, c in body:
            for nr, nc in _neighbors8(r, c):
                if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == BLACK:
                    outline.add((nr, nc))
        pixels = {coord: GREY for coord in body}
        pixels[(marker_r, marker_c)] = marker_color
        for coord in outline:
            pixels[coord] = outline_color
        infos.append(
            {
                "body": body,
                "marker": (marker_r, marker_c, marker_color),
                "outline": outline,
                "pixels": pixels,
            }
        )
    return infos


def _orbit_positions(r: int, c: int, axis_r: int, axis_c: int) -> tuple[tuple[int, int], ...]:
    return (
        (r, c),
        (2 * axis_r - r, c),
        (r, 2 * axis_c - c),
        (2 * axis_r - r, 2 * axis_c - c),
    )


def _paint_outline(out: np.ndarray, sprite_infos: list[dict[str, object]]) -> None:
    for info in sprite_infos:
        marker_color = int(info["marker"][2])  # type: ignore[index]
        outline_color = MARKER_TO_OUTLINE[marker_color]
        for r, c in info["outline"]:  # type: ignore[index]
            assert int(out[r, c]) == BLACK, "outlined regions must be background in the input"
            out[r, c] = outline_color


def _build_combined_orbit(
    pixel_map: dict[tuple[int, int], int],
    axis_r: int,
    axis_c: int,
    shape: tuple[int, int],
) -> dict[tuple[int, int], int] | None:
    combined: dict[tuple[int, int], int] = {}
    for (r, c), color in pixel_map.items():
        for rr, cc in _orbit_positions(r, c, axis_r, axis_c):
            if not _in_bounds(shape, rr, cc):
                return None
            prev = combined.get((rr, cc))
            if prev is not None and prev != color:
                return None
            combined[(rr, cc)] = color
    return combined


def main(grid: np.ndarray) -> np.ndarray:
    """
    Outline each gray sprite using the color implied by its marker, then mirror the outlined sprite across the axes of the separate blue symmetric marker.

    Input structure:
    - A black background contains one blue object with both horizontal and vertical mirror symmetry.
    - The grid also contains one or two irregular 4-connected gray sprites.
    - Each gray sprite has exactly one adjacent colored marker pixel chosen from red, green, or teal.

    Rule:
    - Identify the symmetric blue object and use its center row and center column as reflection axes.
    - For each gray sprite, find its unique marker and map marker color to outline color: green->pink, red->blue, teal->yellow.
    - Paint the sprite's outer 8-neighbor background ring in that outline color, keeping the gray body and the marker unchanged.
    - Reflect every pixel belonging to the body, marker, and outline across the horizontal axis, the vertical axis, and both axes together; draw each reflected copy only onto black cells.
    - Keep the blue symmetric object unchanged.

    Key insight:
    - The marker color determines how each sprite is outlined, while the separate blue symmetric object supplies the off-center reflection axes that replicate the whole outlined sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    axis_r, axis_c, _blue_pixels, _bbox = _find_axis_object(g)
    sprite_infos = _extract_sprite_infos(g)

    out = g.copy()
    _paint_outline(out, sprite_infos)

    for info in sprite_infos:
        combined = _build_combined_orbit(info["pixels"], axis_r, axis_c, g.shape)  # type: ignore[arg-type]
        assert combined is not None, "all mirrored pixels must stay within the grid"
        for (r, c), color in combined.items():
            if int(out[r, c]) == BLACK:
                out[r, c] = color
    return out


def _random_gray_shape(rng: np.random.Generator) -> set[tuple[int, int]]:
    for _ in range(400):
        h = int(rng.integers(2, 5))
        w = int(rng.integers(2, 5))
        size = int(rng.integers(3, min(6, h * w) + 1))
        seed = (int(rng.integers(h)), int(rng.integers(w)))
        cells = {seed}
        while len(cells) < size:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
        if len(cells) != size:
            continue
        cells, _, _ = _normalize_coords(cells)
        mask = _coords_to_mask(cells)
        if _is_hv_symmetric(mask):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        return cells
    raise RuntimeError("failed to generate a non-symmetric grey sprite")


def _make_local_sprite(
    shape_cells: set[tuple[int, int]],
    marker_color: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    candidates = set()
    for r, c in shape_cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) not in shape_cells:
                candidates.add((nr, nc))
    marker = tuple(sorted(candidates)[int(rng.integers(len(candidates)))])

    outline = set()
    for r, c in shape_cells:
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) not in shape_cells and (nr, nc) != marker:
                outline.add((nr, nc))

    all_coords = set(shape_cells) | {marker} | outline
    rows = [r for r, _ in all_coords]
    cols = [c for _, c in all_coords]
    r_shift, c_shift = -min(rows), -min(cols)

    def shift_coords(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return {(r + r_shift, c + c_shift) for r, c in coords}

    body = shift_coords(shape_cells)
    marker_pos = (marker[0] + r_shift, marker[1] + c_shift)
    outline = shift_coords(outline)

    visible = {coord: GREY for coord in body}
    visible[marker_pos] = marker_color
    composite = dict(visible)
    for coord in outline:
        composite[coord] = MARKER_TO_OUTLINE[marker_color]

    h = max(r for r, _ in composite) + 1
    w = max(c for _, c in composite) + 1
    return {
        "visible": visible,
        "composite": composite,
        "shape": (h, w),
    }


def _translate(pixel_map: dict[tuple[int, int], int], r0: int, c0: int) -> dict[tuple[int, int], int]:
    return {(r + r0, c + c0): color for (r, c), color in pixel_map.items()}


def _candidate_positions(
    grid_shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    quad: tuple[str, str],
    sprite_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = grid_shape
    blue_r0, blue_c0, blue_r1, blue_c1 = bbox
    sh, sw = sprite_shape

    if quad[0] == "top":
        r_start, r_end = 0, blue_r0 - sh - 1
    else:
        r_start, r_end = blue_r1 + 2, h - sh

    if quad[1] == "left":
        c_start, c_end = 0, blue_c0 - sw - 1
    else:
        c_start, c_end = blue_c1 + 2, w - sw

    if r_start > r_end or c_start > c_end:
        return []
    return [(r, c) for r in range(r_start, r_end + 1) for c in range(c_start, c_end + 1)]


def _place_sprite(
    grid: np.ndarray,
    reserved: dict[tuple[int, int], int],
    sprite: dict[str, object],
    axis_r: int,
    axis_c: int,
    bbox: tuple[int, int, int, int],
    quad: tuple[str, str],
    rng: np.random.Generator,
) -> bool:
    positions = _candidate_positions(grid.shape, bbox, quad, sprite["shape"])  # type: ignore[arg-type]
    if not positions:
        return False

    order = rng.permutation(len(positions))
    for idx in order:
        r0, c0 = positions[int(idx)]
        visible = _translate(sprite["visible"], r0, c0)  # type: ignore[arg-type]
        composite = _translate(sprite["composite"], r0, c0)  # type: ignore[arg-type]
        orbit = _build_combined_orbit(composite, axis_r, axis_c, grid.shape)
        if orbit is None:
            continue
        if any(coord in reserved for coord in orbit):
            continue
        if any(int(grid[r, c]) != BLACK for r, c in visible):
            continue
        for (r, c), color in visible.items():
            grid[r, c] = color
        reserved.update(orbit)
        return True
    return False


def _pick_grid_and_axis(rng: np.random.Generator) -> tuple[np.ndarray, int, int, dict[tuple[int, int], int], tuple[int, int, int, int]]:
    for _ in range(200):
        h = int(rng.integers(17, 25))
        w = int(rng.integers(17, 25))
        grid = np.full((h, w), BLACK, dtype=int)

        marker_mask = SYMMETRIC_MARKERS[int(rng.integers(len(SYMMETRIC_MARKERS)))]
        mh, mw = marker_mask.shape
        half_r, half_c = mh // 2, mw // 2

        axis_r = int(rng.integers(half_r + 5, h - half_r - 5))
        axis_c = int(rng.integers(half_c + 5, w - half_c - 5))
        if h % 2 == 1 and w % 2 == 1 and axis_r == h // 2 and axis_c == w // 2:
            continue

        top = axis_r - half_r
        left = axis_c - half_c
        blue_pixels: dict[tuple[int, int], int] = {}
        for r in range(mh):
            for c in range(mw):
                if marker_mask[r, c]:
                    coord = (top + r, left + c)
                    blue_pixels[coord] = BLUE
                    grid[coord] = BLUE
        bbox = (top, left, top + mh - 1, left + mw - 1)
        return grid, axis_r, axis_c, blue_pixels, bbox
    raise RuntimeError("failed to place the blue symmetric marker")


def _oracle_accept(grid: np.ndarray) -> bool:
    try:
        axis_r, axis_c, _blue_pixels, _bbox = _find_axis_object(grid)
        sprite_infos = _extract_sprite_infos(grid)
        out = main(grid)
    except Exception:
        return False

    if np.array_equal(out, grid):
        return False

    added = np.count_nonzero(out != grid)
    if added <= 0:
        return False

    for info in sprite_infos:
        rows = [r for r, _ in info["pixels"]]  # type: ignore[index]
        cols = [c for _, c in info["pixels"]]  # type: ignore[index]
        same_row_side = max(rows) < axis_r or min(rows) > axis_r
        same_col_side = max(cols) < axis_c or min(cols) > axis_c
        if not (same_row_side and same_col_side):
            return False

        for (r, c), color in info["pixels"].items():  # type: ignore[index]
            orbit = _orbit_positions(r, c, axis_r, axis_c)
            if len(set(orbit)) != 4:
                return False
            for rr, cc in orbit:
                if not _in_bounds(out.shape, rr, cc) or int(out[rr, cc]) != color:
                    return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    quadrants = (
        ("top", "left"),
        ("top", "right"),
        ("bottom", "left"),
        ("bottom", "right"),
    )

    for _ in range(400):
        grid, axis_r, axis_c, blue_pixels, bbox = _pick_grid_and_axis(rng)
        reserved = dict(blue_pixels)
        sprite_count = 2 if rng.random() < 0.65 else 1
        marker_colors = list(rng.choice(np.array(MARKER_COLORS, dtype=int), size=sprite_count, replace=False))
        quad = quadrants[int(rng.integers(len(quadrants)))]

        sprites = []
        for marker_color in marker_colors:
            shape = _random_gray_shape(rng)
            sprites.append(_make_local_sprite(shape, int(marker_color), rng))
        sprites.sort(key=lambda item: len(item["composite"]), reverse=True)  # type: ignore[arg-type]

        ok = True
        for sprite in sprites:
            if not _place_sprite(grid, reserved, sprite, axis_r, axis_c, bbox, quad, rng):
                ok = False
                break
        if not ok:
            continue
        if _oracle_accept(grid):
            return grid

    raise RuntimeError("failed to generate a valid input for puzzle 11")
