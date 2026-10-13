from __future__ import annotations

import numpy as np


BACKGROUND = 0
MOTIF_COLOR = 5
OBJECT_COLOR_POOL = (1, 2, 3, 4)
PALETTE_COLOR_POOL = (6, 7, 8, 9)

# A 4-object instance would require output width 2 + 4 + 8 + 16 + 3 = 33 > 30,
# so the generator uses 3 objects to stay inside ARC's 30x30 limit.
GEN_OBJECT_COUNT = 3
GEN_HEIGHT_RANGE = (16, 22)
GEN_WIDTH_RANGE = (14, 18)
GEN_MAX_ATTEMPTS = 800
GEN_MIN_L1_GAP = 2
GEN_RESERVED_TOP = 3
GEN_RESERVED_BOTTOM_GAP = 1

MOTIF_PATTERNS = (
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [0, 0]], dtype=bool),
    np.array([[1, 0], [1, 0]], dtype=bool),
    np.array([[0, 1], [0, 1]], dtype=bool),
    np.array([[0, 0], [1, 1]], dtype=bool),
)

OBJECT_DIMS_BY_AREA = {
    4: ((2, 2),),
    6: ((2, 3), (3, 2)),
    8: ((2, 4), (4, 2)),
    9: ((3, 3),),
    10: ((2, 5), (5, 2)),
    12: ((3, 4), (4, 3)),
    15: ((3, 5), (5, 3)),
    16: ((4, 4),),
}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox(arr),
                    "cell_count": int(arr.shape[0]),
                }
            )
    return components


def _component_count(mask: np.ndarray) -> int:
    g = np.asarray(mask, dtype=bool).astype(int)
    return len(_find_components(g, bg=0))


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((factor, factor), dtype=int)).astype(bool)


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, list[int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 4 and w >= 4, "grid is too small"

    bg = _get_bg(g)
    assert bg == BACKGROUND, "background must be black"

    motif = g[:2, :2]
    assert set(map(int, np.unique(motif))).issubset({BACKGROUND, MOTIF_COLOR}), "invalid 2x2 motif colors"
    motif_mask = motif == MOTIF_COLOR
    assert int(motif_mask.sum()) in (2, 3), "motif must use 2-3 grey pixels"
    assert _component_count(motif_mask) == 1, "motif foreground must be 4-connected"

    palette_row = g[h - 1]
    palette_cols = np.flatnonzero(palette_row != bg)
    assert palette_cols.size > 0, "missing bottom palette"
    assert np.array_equal(palette_cols, np.arange(int(palette_cols[0]), int(palette_cols[-1]) + 1)), "palette must be contiguous"
    palette = [int(palette_row[c]) for c in palette_cols]
    assert len(set(palette)) == len(palette), "palette colors must be unique"
    assert set(palette).issubset(PALETTE_COLOR_POOL), "invalid palette colors"
    assert np.all(palette_row[: int(palette_cols[0])] == bg) and np.all(palette_row[int(palette_cols[-1]) + 1 :] == bg)

    stripped = g.copy()
    stripped[:2, :2] = bg
    stripped[h - 1, palette_cols] = bg
    components = _find_components(stripped, bg)
    assert len(components) == len(palette), "palette and object counts must match"
    assert len(set(component["color"] for component in components)) == len(components), "objects must use distinct colors"
    assert set(component["color"] for component in components).issubset(OBJECT_COLOR_POOL), "invalid object colors"

    areas = [_bbox_area(component["bbox"]) for component in components]
    assert len(set(areas)) == len(areas), "bbox areas must be unique"
    ordered = sorted(components, key=lambda component: _bbox_area(component["bbox"]))
    return motif_mask, palette, ordered


def _build_output(motif_mask: np.ndarray, palette: list[int]) -> np.ndarray:
    count = len(palette)
    scales = [1 << k for k in range(count)]
    height = 2 * scales[-1]
    width = sum(2 * scale for scale in scales) + (count - 1)
    assert height <= 30 and width <= 30, "output exceeds ARC size limit"

    out = np.zeros((height, width), dtype=int)
    c0 = 0
    for scale, color in zip(scales, palette):
        stamp_mask = _scale_mask(motif_mask, scale)
        stamp = np.zeros((2 * scale, 2 * scale), dtype=int)
        stamp[stamp_mask] = int(color)
        r0 = height - stamp.shape[0]
        out[r0 : r0 + stamp.shape[0], c0 : c0 + stamp.shape[1]] = stamp
        c0 += stamp.shape[1] + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Sort token objects by bbox area, assign dyadic motif scales, recolor them from the palette, and bottom-stack them left-to-right.

    Input structure:
    - The background is black.
    - The top-left 2x2 patch is a grey motif containing 2-3 foreground pixels and 1-2 background pixels.
    - Elsewhere are 3-4 scattered monochrome 4-connected objects in distinct colors, plus one contiguous palette row on the bottom edge.

    Rule:
    - Read the top-left 2x2 region as motif M and the contiguous non-background segment on the bottom row as palette P.
    - Remove those structures, then sort the remaining monochrome objects by increasing bbox area.
    - For the k-th object in that order, scale M by 2^k, recolor its foreground pixels with P[k], keep motif-background pixels black, and place the result in the output.
    - Pack the recolored scaled motifs from left to right with one black column between them and align their bottoms.

    Key insight:
    - The scattered objects are only ordering tokens: their bbox areas determine which palette color receives which dyadic motif scale, while their own shapes and positions disappear from the output.
    """
    motif_mask, palette, _ordered = _parse_input(np.asarray(grid, dtype=int))
    return _build_output(motif_mask, palette)


def _random_motif_mask() -> np.ndarray:
    return MOTIF_PATTERNS[int(np.random.randint(len(MOTIF_PATTERNS)))]


def _random_monotone_path(h: int, w: int) -> set[tuple[int, int]]:
    moves = [0] * (h - 1) + [1] * (w - 1)
    np.random.shuffle(moves)
    r = c = 0
    cells = {(0, 0)}
    for move in moves:
        if move == 0:
            r += 1
        else:
            c += 1
        cells.add((r, c))
    return cells


def _frontier(cells: set[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                result.add((nr, nc))
    return list(result)


def _flip_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out = set(cells)
    if np.random.rand() < 0.5:
        out = {(r, w - 1 - c) for r, c in out}
    if np.random.rand() < 0.5:
        out = {(h - 1 - r, c) for r, c in out}
    return out


def _make_object_patch(h: int, w: int, color: int) -> tuple[np.ndarray, int]:
    area = h * w
    min_cells = h + w - 1
    max_cells = area - 1
    assert min_cells <= max_cells, "chosen bbox cannot host a non-solid spanning shape"
    target = int(np.random.randint(min_cells, max_cells + 1))
    cells = _random_monotone_path(h, w)
    while len(cells) < target:
        choices = _frontier(cells, h, w)
        cells.add(choices[int(np.random.randint(len(choices)))])
    cells = _flip_cells(cells, h, w)

    patch = np.zeros((h, w), dtype=int)
    for r, c in cells:
        patch[r, c] = int(color)
    return patch, len(cells)


def _coords_from_patch(patch: np.ndarray, r0: int, c0: int) -> np.ndarray:
    coords = np.argwhere(np.asarray(patch, dtype=int) != 0)
    coords[:, 0] += int(r0)
    coords[:, 1] += int(c0)
    return coords


def _far_enough(coords: np.ndarray, existing: list[np.ndarray], forbidden: set[tuple[int, int]]) -> bool:
    for r, c in coords:
        if (int(r), int(c)) in forbidden:
            return False
    if not existing:
        return True
    other = np.vstack(existing)
    distances = np.abs(coords[:, None, :] - other[None, :, :]).sum(axis=2)
    return bool(np.all(distances >= GEN_MIN_L1_GAP))


def _build_input_grid(
    height: int,
    width: int,
    motif_mask: np.ndarray,
    palette: list[int],
    specs: list[dict],
    placements: list[tuple[int, int]],
    palette_start: int,
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    motif = np.zeros((2, 2), dtype=int)
    motif[motif_mask] = MOTIF_COLOR
    grid[:2, :2] = motif
    for spec, (r0, c0) in zip(specs, placements):
        patch = spec["patch"]
        mask = patch != 0
        grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]
    for i, color in enumerate(palette):
        grid[height - 1, palette_start + i] = int(color)
    return grid


def _rank(values: list[int]) -> tuple[int, ...]:
    return tuple(int(i) for i in np.argsort(np.array(values)))


def _sample_specs() -> list[dict] | None:
    areas = list(OBJECT_DIMS_BY_AREA)
    chosen_areas = list(np.random.choice(np.array(areas, dtype=int), size=GEN_OBJECT_COUNT, replace=False))
    colors = list(np.random.choice(np.array(OBJECT_COLOR_POOL, dtype=int), size=GEN_OBJECT_COUNT, replace=False))
    specs: list[dict] = []
    for color, area in zip(colors, chosen_areas):
        dims = OBJECT_DIMS_BY_AREA[int(area)]
        h, w = dims[int(np.random.randint(len(dims)))]
        patch, cell_count = _make_object_patch(h, w, int(color))
        specs.append(
            {
                "color": int(color),
                "patch": patch,
                "bbox_area": int(area),
                "cell_count": int(cell_count),
            }
        )

    counts = [spec["cell_count"] for spec in specs]
    if len(set(counts)) != len(counts):
        return None
    area_order = _rank([spec["bbox_area"] for spec in specs])
    count_order = _rank(counts)
    color_order = _rank([spec["color"] for spec in specs])
    if count_order == area_order or color_order == area_order:
        return None
    return specs


def _sample_placements(height: int, width: int, specs: list[dict]) -> list[tuple[int, int]] | None:
    forbidden = {(r, c) for r in range(GEN_RESERVED_TOP) for c in range(GEN_RESERVED_TOP)}
    bottom_limit = height - 2 - GEN_RESERVED_BOTTOM_GAP
    order = sorted(range(len(specs)), key=lambda i: specs[i]["patch"].shape[0] * specs[i]["patch"].shape[1], reverse=True)
    placed_coords: list[np.ndarray] = []
    placements: list[tuple[int, int] | None] = [None] * len(specs)

    for idx in order:
        patch = specs[idx]["patch"]
        h, w = patch.shape
        choices: list[tuple[int, int, np.ndarray]] = []
        for r0 in range(0, bottom_limit - h + 2):
            for c0 in range(0, width - w + 1):
                coords = _coords_from_patch(patch, r0, c0)
                if _far_enough(coords, placed_coords, forbidden):
                    choices.append((r0, c0, coords))
        if not choices:
            return None
        choice = choices[int(np.random.randint(len(choices)))]
        placements[idx] = (int(choice[0]), int(choice[1]))
        placed_coords.append(choice[2])

    assert all(position is not None for position in placements)
    return [(int(r0), int(c0)) for r0, c0 in placements]  # type: ignore[misc]


def _validate_semantics(
    grid: np.ndarray,
    motif_mask: np.ndarray,
    palette: list[int],
    specs: list[dict],
    placements: list[tuple[int, int]],
) -> bool:
    parsed_motif, parsed_palette, parsed_components = _parse_input(grid)
    if not np.array_equal(parsed_motif, motif_mask):
        return False
    if parsed_palette != palette:
        return False
    if sorted(_bbox_area(component["bbox"]) for component in parsed_components) != sorted(spec["bbox_area"] for spec in specs):
        return False

    areas = [spec["bbox_area"] for spec in specs]
    counts = [spec["cell_count"] for spec in specs]
    colors = [spec["color"] for spec in specs]
    lefts = [c0 for _r0, c0 in placements]
    tops = [r0 for r0, _c0 in placements]

    if len(set(lefts)) != len(lefts) or len(set(tops)) != len(tops):
        return False

    area_order = _rank(areas)
    if _rank(counts) == area_order:
        return False
    if _rank(colors) == area_order:
        return False
    if _rank(lefts) == area_order:
        return False
    if _rank(tops) == area_order:
        return False

    expected = _build_output(motif_mask, palette)
    actual = main(grid)
    return np.array_equal(actual, expected)


def _generate_candidate() -> np.ndarray | None:
    height = int(np.random.randint(GEN_HEIGHT_RANGE[0], GEN_HEIGHT_RANGE[1] + 1))
    width = int(np.random.randint(GEN_WIDTH_RANGE[0], GEN_WIDTH_RANGE[1] + 1))
    motif_mask = _random_motif_mask()
    palette = [int(c) for c in np.random.choice(np.array(PALETTE_COLOR_POOL, dtype=int), size=GEN_OBJECT_COUNT, replace=False)]
    palette_start = int(np.random.randint(0, width - GEN_OBJECT_COUNT + 1))

    specs = _sample_specs()
    if specs is None:
        return None
    placements = _sample_placements(height, width, specs)
    if placements is None:
        return None

    grid = _build_input_grid(height, width, motif_mask, palette, specs, placements, palette_start)
    if not _validate_semantics(grid, motif_mask, palette, specs, placements):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
