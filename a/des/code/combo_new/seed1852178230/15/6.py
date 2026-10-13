from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
MAROON = 9
OBJECT_COLORS = (1, 2, 3, 5, 6, 7, 8)

MIN_REGIONS = 2
MAX_REGIONS = 5
MIN_REGION_HEIGHT = 3
MAX_REGION_HEIGHT = 6
WIDTH_OPTIONS = tuple(range(10, 21, 2))
GEN_RETRY_LIMIT = 400


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _unique_orientations(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for k in range(4):
            rotated = _crop_mask(np.rot90(base, k))
            key = (tuple(rotated.shape), rotated.astype(np.uint8).tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(rotated)
    return variants


def _build_shape_catalog() -> tuple[np.ndarray, ...]:
    bases = [
        np.array([[1, 1]], dtype=bool),
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1], [1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    ]
    shapes: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for base in bases:
        for variant in _unique_orientations(base):
            area = int(variant.sum())
            if not 2 <= area <= 5:
                continue
            key = (tuple(variant.shape), variant.astype(np.uint8).tobytes())
            if key not in seen:
                seen.add(key)
                shapes.append(variant)
    return tuple(shapes)


SHAPES = _build_shape_catalog()


def _is_separator_row(row: np.ndarray) -> bool:
    return bool(
        np.all((row == MAROON) | (row == YELLOW))
        and np.count_nonzero(row == YELLOW) <= 1
        and np.count_nonzero(row == MAROON) >= row.size - 1
    )


def _paint_shape(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    view = grid[top : top + h, left : left + w]
    view[mask] = int(color)


def _sample_region_heights() -> list[int]:
    for _ in range(200):
        n_regions = random.randint(MIN_REGIONS, MAX_REGIONS)
        heights = [random.randint(MIN_REGION_HEIGHT, MAX_REGION_HEIGHT) for _ in range(n_regions)]
        if sum(heights) + n_regions + 1 <= 30:
            return heights
    raise RuntimeError("failed to sample region heights")


def _candidate_positions(
    region_h: int,
    width: int,
    shape_h: int,
    shape_w: int,
    target_r: int,
    target_c: int,
) -> list[tuple[int, int]]:
    max_r = region_h - shape_h
    max_c = width - shape_w
    positions = [
        (r, c)
        for r in range(max_r + 1)
        for c in range(max_c + 1)
        if (r, c) != (target_r, target_c)
    ]
    if not positions:
        return []

    interior = [
        (r, c)
        for r, c in positions
        if 0 < r < max_r and 0 < c < max_c
    ]
    if interior:
        return interior

    non_corner = [
        (r, c)
        for r, c in positions
        if not (r in (0, max_r) and c in (0, max_c))
    ]
    return non_corner or positions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each monochrome object inside its horizontal band to the corner encoded by a yellow boundary marker.

    Input structure:
    - The background is black.
    - Full-width horizontal maroon separator rows partition the grid into several horizontal bands.
    - Each band contains exactly one small monochrome connected object in its interior.
    - Some separator rows contain a single yellow pixel instead of maroon; for any band, exactly one of its two boundary rows is marked.

    Rule:
    - For each band, inspect whether the yellow marker is on the top boundary or bottom boundary to choose top vs. bottom anchoring.
    - Read the marker column: a marker in the left half means anchor left, and a marker in the right half means anchor right.
    - Translate the band's object without rotating it so that the corresponding corner of its bounding box matches the corresponding interior corner of the band.
    - Clear the object's old cells to black, and keep all maroon separator pixels and yellow marker pixels unchanged.

    Key insight:
    - The yellow marker is a two-bit spatial code: boundary row gives the vertical choice, while left-half vs. right-half gives the horizontal choice.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = map(int, g.shape)

    separator_rows = [r for r in range(height) if _is_separator_row(g[r])]
    assert len(separator_rows) >= 2, "expected at least two separator rows"

    out = g.copy()
    for top_sep, bottom_sep in zip(separator_rows, separator_rows[1:]):
        assert bottom_sep - top_sep >= 2, "expected non-empty band interior"
        band = g[top_sep + 1 : bottom_sep, :]
        band_coords = np.argwhere(band != BLACK)
        assert band_coords.size > 0, "expected one foreground object per band"

        band_colors = np.unique(band[band != BLACK])
        assert band_colors.size == 1, "expected a monochrome object in each band"
        color = int(band_colors[0])

        top_markers = np.where(g[top_sep] == YELLOW)[0]
        bottom_markers = np.where(g[bottom_sep] == YELLOW)[0]
        assert len(top_markers) + len(bottom_markers) == 1, "expected exactly one marker across the two boundaries"

        if len(top_markers) == 1:
            marker_col = int(top_markers[0])
            anchor_top = True
        else:
            marker_col = int(bottom_markers[0])
            anchor_top = False
        anchor_left = marker_col * 2 < width

        r0 = int(band_coords[:, 0].min())
        r1 = int(band_coords[:, 0].max())
        c0 = int(band_coords[:, 1].min())
        c1 = int(band_coords[:, 1].max())
        bbox_h = r1 - r0 + 1
        bbox_w = c1 - c0 + 1

        abs_coords = band_coords + np.array([top_sep + 1, 0], dtype=int)
        out[abs_coords[:, 0], abs_coords[:, 1]] = BLACK

        target_top = top_sep + 1 if anchor_top else bottom_sep - bbox_h
        target_left = 0 if anchor_left else width - bbox_w
        rel_coords = band_coords - np.array([r0, c0], dtype=int)
        out[target_top + rel_coords[:, 0], target_left + rel_coords[:, 1]] = color

    return out


def _gen_attempt() -> np.ndarray | None:
    heights = _sample_region_heights()
    width = random.choice(WIDTH_OPTIONS)
    n_regions = len(heights)

    sep_rows = [0]
    for band_h in heights:
        sep_rows.append(sep_rows[-1] + band_h + 1)
    total_height = sep_rows[-1] + 1

    inp = np.full((total_height, width), BLACK, dtype=int)
    out = np.full((total_height, width), BLACK, dtype=int)

    for row in sep_rows:
        inp[row, :] = MAROON
        out[row, :] = MAROON

    offset = random.randint(0, 1)
    marked_flags = [((idx + offset) % 2 == 0) for idx in range(n_regions + 1)]
    side_by_sep: dict[int, str] = {}
    half = width // 2
    for sep_idx, row in enumerate(sep_rows):
        if not marked_flags[sep_idx]:
            continue
        side = random.choice(("left", "right"))
        if side == "left":
            col = random.randint(0, half - 1)
        else:
            col = random.randint(half, width - 1)
        inp[row, col] = YELLOW
        out[row, col] = YELLOW
        side_by_sep[row] = side

    colors = random.sample(list(OBJECT_COLORS), n_regions)
    for region_idx, band_h in enumerate(heights):
        top_sep = sep_rows[region_idx]
        bottom_sep = sep_rows[region_idx + 1]
        top_marked = marked_flags[region_idx]
        bottom_marked = marked_flags[region_idx + 1]
        assert top_marked ^ bottom_marked

        marked_row = top_sep if top_marked else bottom_sep
        side = side_by_sep[marked_row]
        color = colors[region_idx]

        fitting_shapes = [shape for shape in SHAPES if shape.shape[0] <= band_h and shape.shape[1] <= width]
        random.shuffle(fitting_shapes)
        placed = False
        for shape in fitting_shapes:
            shape_h, shape_w = map(int, shape.shape)
            target_r = 0 if top_marked else band_h - shape_h
            target_c = 0 if side == "left" else width - shape_w
            positions = _candidate_positions(band_h, width, shape_h, shape_w, target_r, target_c)
            if not positions:
                continue
            src_r, src_c = random.choice(positions)
            abs_top = top_sep + 1
            _paint_shape(inp, shape, abs_top + src_r, src_c, color)
            _paint_shape(out, shape, abs_top + target_r, target_c, color)
            placed = True
            break
        if not placed:
            return None

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid band-anchor input")
