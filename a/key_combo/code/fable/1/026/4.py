import numpy as np


BLACK = 0
RED = 2
PALETTE = (1, 3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300

PARTITION_SHAPES = (
    (1, 3),
    (1, 4),
    (2, 2),
    (2, 3),
    (3, 1),
    (3, 2),
)
PARTITION_WEIGHTS = np.array([1, 1, 3, 3, 1, 2], dtype=float)

ROW_SIZE_RANGE = (5, 10)  # upper bound exclusive
COL_SIZE_RANGE = (5, 11)  # upper bound exclusive
TILE_MIN_SIZE = 2
TILE_MAX_SIZE = 5


def bbox(mask: np.ndarray):
    """Return the inclusive bbox of a boolean mask, or None when empty."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box):
    """Crop a grid using an inclusive bbox."""
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _spans_from_dividers(length: int, dividers: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for cut in sorted(dividers):
        if start <= cut - 1:
            spans.append((start, cut - 1))
        start = cut + 1
    if start <= length - 1:
        spans.append((start, length - 1))
    return spans


def _partition_spans(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int]]:
    spans: list[tuple[int, int]] = []
    dividers: list[int] = []
    cursor = 0
    for i, size in enumerate(sizes):
        start = cursor
        end = start + int(size) - 1
        spans.append((start, end))
        cursor = end + 1
        if i != len(sizes) - 1:
            dividers.append(cursor)
            cursor += 1
    return spans, dividers


def _tile_fill(shape: tuple[int, int], tile: np.ndarray) -> np.ndarray:
    h, w = map(int, shape)
    th, tw = map(int, tile.shape)
    rr = np.arange(h)[:, None] % th
    cc = np.arange(w)[None, :] % tw
    return tile[rr, cc]


def _negative_tile(tile: np.ndarray, color: int) -> np.ndarray:
    values = set(map(int, np.unique(tile)))
    assert values <= {BLACK, int(color)}
    return np.where(tile == BLACK, int(color), BLACK).astype(int)


def _extract_regions(grid: np.ndarray) -> tuple[list[dict], list[int], list[int]]:
    h, w = map(int, grid.shape)
    divider_rows = [r for r in range(h) if np.all(grid[r, :] == RED)]
    divider_cols = [c for c in range(w) if np.all(grid[:, c] == RED)]
    row_spans = _spans_from_dividers(h, divider_rows)
    col_spans = _spans_from_dividers(w, divider_cols)
    assert row_spans and col_spans

    regions: list[dict] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = grid[r0 : r1 + 1, c0 : c1 + 1]
            colors = set(map(int, np.unique(region))) - {BLACK}
            assert RED not in colors
            assert len(colors) == 1
            color = colors.pop()
            seed_box = bbox(region == color)
            assert seed_box is not None
            sr0, sc0, sr1, sc1 = seed_box
            assert sr0 == 0 and sc0 == 0
            tile = crop_bbox(region, seed_box)
            assert np.any(tile == color)
            assert np.any(tile == BLACK)
            regions.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "shape": (r1 - r0 + 1, c1 - c0 + 1),
                    "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                    "color": int(color),
                    "seed": tile,
                }
            )
    assert len(regions) >= 2
    return regions, divider_rows, divider_cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile every divider-defined rectangle with its seed, except the unique smallest and largest rectangles, which use each other's negative seed.

    Input structure:
    - Black cells form several rectangular regions separated by full red divider rows and columns.
    - Each region contains exactly one non-black, non-red color, visible only inside a small seed tile anchored at that region's top-left corner.
    - Inside the seed tile's bbox, colored cells and black cells coexist; the rest of the region is black.

    Rule:
    - Split the grid along the full red divider rows and columns into rectangular regions.
    - For each region, recover its seed tile as the bbox of its unique colored cells and tile that patch across the whole region from the region's top-left corner, clipping at the borders.
    - Let the unique smallest-area region and unique largest-area region be the two extremes.
    - Normal regions use their own seed. The smallest region instead uses the largest region's seed negative, and the largest region uses the smallest region's seed negative. A negative swaps black and colored cells inside the donor seed while keeping the donor seed's color. Red divider lines stay unchanged.

    Key insight:
    - The divider grid only identifies candidate rectangles; the actual twist is that area extremes exchange complemented seed masks rather than swapping the visible seeds directly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    regions, divider_rows, divider_cols = _extract_regions(g)
    areas = [region["area"] for region in regions]
    min_area = min(areas)
    max_area = max(areas)
    assert areas.count(min_area) == 1
    assert areas.count(max_area) == 1
    min_idx = areas.index(min_area)
    max_idx = areas.index(max_area)

    out = np.full(g.shape, BLACK, dtype=int)
    out[divider_rows, :] = RED
    out[:, divider_cols] = RED

    min_donor = regions[max_idx]
    max_donor = regions[min_idx]
    min_tile = _negative_tile(min_donor["seed"], min_donor["color"])
    max_tile = _negative_tile(max_donor["seed"], max_donor["color"])

    for idx, region in enumerate(regions):
        if idx == min_idx:
            tile = min_tile
        elif idx == max_idx:
            tile = max_tile
        else:
            tile = region["seed"]
        patch = _tile_fill(region["shape"], tile)
        r0, c0, r1, c1 = region["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = patch

    out[divider_rows, :] = RED
    out[:, divider_cols] = RED
    return out


def _primitive_mask(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            if ph == h and pw == w:
                continue
            candidate = _tile_fill((h, w), mask[:ph, :pw].astype(int)).astype(bool)
            if np.array_equal(candidate, mask):
                return False
    return True


def _random_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray | None:
    for _ in range(80):
        density = float(rng.uniform(0.35, 0.65))
        mask = rng.random((h, w)) < density
        if int(mask.sum()) < 3 or int((~mask).sum()) < 2:
            continue
        if not mask[0, :].any():
            continue
        if not mask[-1, :].any():
            continue
        if not mask[:, 0].any():
            continue
        if not mask[:, -1].any():
            continue
        if not _primitive_mask(mask):
            continue
        return mask
    return None


def _build_region_specs(
    rng: np.random.Generator,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> list[dict] | None:
    positions = [(ri, ci) for ri in range(len(row_spans)) for ci in range(len(col_spans))]
    colors = rng.choice(np.array(PALETTE, dtype=int), size=len(positions), replace=False)
    specs: list[dict] = []
    for idx, (ri, ci) in enumerate(positions):
        r0, r1 = row_spans[ri]
        c0, c1 = col_spans[ci]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        max_th = min(TILE_MAX_SIZE, height - 1)
        max_tw = min(TILE_MAX_SIZE, width - 1)
        if max_th < TILE_MIN_SIZE or max_tw < TILE_MIN_SIZE:
            return None
        tile_h = int(rng.integers(TILE_MIN_SIZE, max_th + 1))
        tile_w = int(rng.integers(TILE_MIN_SIZE, max_tw + 1))
        mask = _random_mask(rng, tile_h, tile_w)
        if mask is None:
            return None
        color = int(colors[idx])
        tile = np.where(mask, color, BLACK).astype(int)
        specs.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": (height, width),
                "area": int(height * width),
                "color": color,
                "seed": tile,
            }
        )
    return specs


def _extreme_regions_are_non_degenerate(specs: list[dict]) -> bool:
    areas = [spec["area"] for spec in specs]
    min_area = min(areas)
    max_area = max(areas)
    if areas.count(min_area) != 1 or areas.count(max_area) != 1:
        return False

    min_idx = areas.index(min_area)
    max_idx = areas.index(max_area)
    min_spec = specs[min_idx]
    max_spec = specs[max_idx]

    min_expected = _tile_fill(min_spec["shape"], _negative_tile(max_spec["seed"], max_spec["color"]))
    max_expected = _tile_fill(max_spec["shape"], _negative_tile(min_spec["seed"], min_spec["color"]))
    min_direct = _tile_fill(min_spec["shape"], max_spec["seed"])
    max_direct = _tile_fill(max_spec["shape"], min_spec["seed"])
    min_self = _tile_fill(min_spec["shape"], min_spec["seed"])
    max_self = _tile_fill(max_spec["shape"], max_spec["seed"])

    if np.array_equal(min_expected, min_direct) or np.array_equal(max_expected, max_direct):
        return False
    if np.array_equal(min_expected, min_self) or np.array_equal(max_expected, max_self):
        return False
    return True


def _render_input(
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    divider_rows: list[int],
    divider_cols: list[int],
    specs: list[dict],
) -> np.ndarray:
    h = row_spans[-1][1] + 1
    w = col_spans[-1][1] + 1
    grid = np.full((h, w), BLACK, dtype=int)
    grid[divider_rows, :] = RED
    grid[:, divider_cols] = RED

    for spec in specs:
        r0, c0, _, _ = spec["bbox"]
        tile = spec["seed"]
        th, tw = tile.shape
        patch = grid[r0 : r0 + th, c0 : c0 + tw]
        patch[tile != BLACK] = tile[tile != BLACK]
    return grid


def _expected_from_specs(
    shape: tuple[int, int],
    divider_rows: list[int],
    divider_cols: list[int],
    specs: list[dict],
) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    out[divider_rows, :] = RED
    out[:, divider_cols] = RED

    areas = [spec["area"] for spec in specs]
    min_idx = areas.index(min(areas))
    max_idx = areas.index(max(areas))
    min_donor = specs[max_idx]
    max_donor = specs[min_idx]

    for idx, spec in enumerate(specs):
        if idx == min_idx:
            tile = _negative_tile(min_donor["seed"], min_donor["color"])
        elif idx == max_idx:
            tile = _negative_tile(max_donor["seed"], max_donor["color"])
        else:
            tile = spec["seed"]
        patch = _tile_fill(spec["shape"], tile)
        r0, c0, r1, c1 = spec["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = patch

    out[divider_rows, :] = RED
    out[:, divider_cols] = RED
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    idx = int(rng.choice(len(PARTITION_SHAPES), p=PARTITION_WEIGHTS / PARTITION_WEIGHTS.sum()))
    n_rows, n_cols = PARTITION_SHAPES[idx]

    row_sizes = [int(x) for x in rng.integers(ROW_SIZE_RANGE[0], ROW_SIZE_RANGE[1], size=n_rows)]
    col_sizes = [int(x) for x in rng.integers(COL_SIZE_RANGE[0], COL_SIZE_RANGE[1], size=n_cols)]

    row_spans_inner, divider_rows = _partition_spans(row_sizes)
    col_spans_inner, divider_cols = _partition_spans(col_sizes)
    total_h = row_spans_inner[-1][1] + 1
    total_w = col_spans_inner[-1][1] + 1
    if total_h > 30 or total_w > 30:
        return None

    areas = [rh * cw for rh in row_sizes for cw in col_sizes]
    if areas.count(min(areas)) != 1 or areas.count(max(areas)) != 1:
        return None

    specs = _build_region_specs(rng, row_spans_inner, col_spans_inner)
    if specs is None:
        return None
    if not _extreme_regions_are_non_degenerate(specs):
        return None

    grid = _render_input(row_spans_inner, col_spans_inner, divider_rows, divider_cols, specs)
    expected = _expected_from_specs(grid.shape, divider_rows, divider_cols, specs)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    parsed_specs, parsed_div_rows, parsed_div_cols = _extract_regions(grid)
    parsed_areas = [spec["area"] for spec in parsed_specs]
    if parsed_areas.count(min(parsed_areas)) != 1 or parsed_areas.count(max(parsed_areas)) != 1:
        return None
    if parsed_div_rows != divider_rows or parsed_div_cols != divider_cols:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid extreme-region-negative-tiling input")
