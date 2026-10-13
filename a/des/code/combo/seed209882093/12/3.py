from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8

TEST_COLORS = (BLUE, RED, GREEN, ORANGE, MAGENTA)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_BAND_COUNT_RANGE = (2, 4)
GEN_BAND_HEIGHT_RANGE = (6, 10)
GEN_TEST_COUNT_RANGE = (2, 4)
GEN_MARGIN_RANGE = (1, 3)
GEN_GAP_RANGE = (2, 5)
GEN_SHAPE_ATTEMPTS = 200
GEN_ATTEMPTS = 400


def _normalize_shape(coords: list[tuple[int, int]] | set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(list(coords), dtype=int)
    assert arr.size, "expected a nonempty shape"
    arr = arr - arr.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in arr))


def _shape_size(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _c in shape]
    cols = [c for _r, c in shape]
    return max(rows) + 1, max(cols) + 1


def _rotate_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h, _w = _shape_size(shape)
    return _normalize_shape([(c, h - 1 - r) for r, c in shape])


def _reflect_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    _h, w = _shape_size(shape)
    return _normalize_shape([(r, w - 1 - c) for r, c in shape])


def _rotation_signatures(shape: tuple[tuple[int, int], ...]) -> set[tuple[tuple[int, int], ...]]:
    cur = _normalize_shape(shape)
    variants: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(4):
        variants.add(cur)
        cur = _rotate_shape(cur)
    return variants


def _shape_mask(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = _shape_size(shape)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _has_hole(shape: tuple[tuple[int, int], ...]) -> bool:
    mask = _shape_mask(shape)
    bg = ~mask
    if not bool(bg.any()):
        return False

    h, w = mask.shape
    seen = np.zeros_like(bg, dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))

    return bool((bg & ~seen).any())


def _components(mask: np.ndarray) -> list[dict]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((int(r), int(c)))
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [r for r, _c in cells]
            cols = [c for _r, c in cells]
            comps.append(
                {
                    "coords": tuple(sorted(cells)),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _band_spans(g: np.ndarray) -> list[tuple[int, int]]:
    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == AZURE)]
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in sep_rows:
        assert start < sep, "separator rows must leave nonempty bands"
        spans.append((start, sep))
        start = sep + 1
    assert start < g.shape[0], "expected a final nonempty band"
    spans.append((start, g.shape[0]))
    heights = {r1 - r0 for r0, r1 in spans}
    assert len(heights) == 1, "bands must have equal height"
    return spans


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each test shape by whether it matches the left template up to rotation.

    Input structure:
    - The background is black and full-width azure rows split the grid into equal-height bands.
    - In each band, the leftmost connected component is a two-color template made of grey
      and yellow pixels.
    - To the right are 2-4 single-color test shapes.

    Rule:
    - Compare every test shape with the template using only occupied cells.
    - Rotations by 0/90/180/270 degrees are allowed; flips are not.
    - Matching tests become green; non-matching tests become red.
    - The template and separator rows stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    out = g.copy()
    for r0, r1 in _band_spans(g):
        band = g[r0:r1, :]
        mask = (band != BLACK) & (band != AZURE)
        objects = _components(mask)
        assert len(objects) >= 3, "each band needs one template and at least two tests"
        objects.sort(key=lambda obj: (obj["bbox"][1], obj["bbox"][0]))

        template = objects[0]
        template_colors = {int(band[r, c]) for r, c in template["coords"]}
        assert template_colors <= {GRAY, YELLOW} and YELLOW in template_colors and GRAY in template_colors
        template_shape = _normalize_shape(template["coords"])
        template_rotations = _rotation_signatures(template_shape)

        for obj in objects[1:]:
            obj_colors = {int(band[r, c]) for r, c in obj["coords"]}
            assert len(obj_colors) == 1, "test shapes must be monochrome"
            target_color = GREEN if _normalize_shape(obj["coords"]) in template_rotations else RED
            for rr, cc in obj["coords"]:
                out[r0 + rr, cc] = target_color

    return out


def _random_connected_shape(
    max_dim: int,
    min_area: int = 5,
    max_area: int = 8,
    target_area: int | None = None,
    require_chiral: bool | None = None,
) -> tuple[tuple[int, int], ...] | None:
    max_area = min(max_area, max_dim * max_dim - 1)
    if max_area < min_area:
        return None

    for _ in range(GEN_SHAPE_ATTEMPTS):
        area = int(target_area) if target_area is not None else random.randint(min_area, max_area)
        dims = [(h, w) for h in range(2, max_dim + 1) for w in range(2, max_dim + 1) if h * w > area]
        if not dims:
            return None
        h, w = random.choice(dims)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != area:
            continue

        shape = _normalize_shape(cells)
        sh, sw = _shape_size(shape)
        if sh < 2 or sw < 2:
            continue
        if len(shape) == sh * sw:
            continue
        if _has_hole(shape):
            continue

        rotations = _rotation_signatures(shape)
        if len(rotations) < 2:
            continue
        chiral = _reflect_shape(shape) not in rotations
        if require_chiral is True and not chiral:
            continue
        return shape
    return None


def _sample_top(band_height: int, shape_height: int) -> int:
    slack = band_height - shape_height
    assert slack >= 0
    if slack <= 1:
        return random.randint(0, slack)
    return random.randint(1, slack - 1)


def _render_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], top: int, left: int, color: int) -> None:
    for r, c in shape:
        grid[top + r, left + c] = int(color)


def _render_template(
    grid: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    top: int,
    left: int,
    yellow_cells: set[tuple[int, int]],
) -> None:
    for r, c in shape:
        grid[top + r, left + c] = YELLOW if (r, c) in yellow_cells else GRAY


def _nonmatching_shape(template_shape: tuple[tuple[int, int], ...], max_dim: int) -> tuple[tuple[int, int], ...] | None:
    template_rotations = _rotation_signatures(template_shape)
    reflected = _reflect_shape(template_shape)
    if reflected not in template_rotations and random.random() < 0.6:
        return reflected

    same_area = random.random() < 0.7
    target_area = len(template_shape) if same_area else None
    min_area = max(5, len(template_shape) - 2)
    max_area = min(max_dim * max_dim - 1, len(template_shape) + 2)
    if max_area < min_area:
        return reflected if reflected not in template_rotations else None

    for _ in range(GEN_SHAPE_ATTEMPTS):
        shape = _random_connected_shape(
            max_dim=max_dim,
            min_area=min_area,
            max_area=max_area,
            target_area=target_area,
            require_chiral=None,
        )
        if shape is None:
            continue
        if shape not in template_rotations:
            return shape

    if reflected not in template_rotations:
        return reflected
    return None


def _build_band(band_height: int) -> tuple[np.ndarray, np.ndarray] | None:
    n_tests = random.randint(*GEN_TEST_COUNT_RANGE)
    dim_cap = random.randint(4, min(6, band_height - 1))
    require_chiral = random.random() < 0.6

    base_shape = _random_connected_shape(
        max_dim=dim_cap,
        min_area=5,
        max_area=min(8, dim_cap * dim_cap - 1),
        require_chiral=require_chiral,
    )
    if base_shape is None and require_chiral:
        base_shape = _random_connected_shape(
            max_dim=dim_cap,
            min_area=5,
            max_area=min(8, dim_cap * dim_cap - 1),
            require_chiral=None,
        )
    if base_shape is None:
        return None

    template_shape = random.choice(tuple(_rotation_signatures(base_shape)))
    template_rotations = tuple(_rotation_signatures(template_shape))

    match_count = random.randint(1, n_tests - 1)
    flags = [True] * match_count + [False] * (n_tests - match_count)
    random.shuffle(flags)

    test_shapes: list[tuple[tuple[int, int], ...]] = []
    for is_match in flags:
        if is_match:
            test_shapes.append(random.choice(template_rotations))
        else:
            mismatch = _nonmatching_shape(template_shape, dim_cap)
            if mismatch is None:
                return None
            test_shapes.append(mismatch)

    colors = random.sample(TEST_COLORS, n_tests)
    yellow_count = random.randint(1, min(3, len(template_shape) - 1))
    yellow_cells = set(random.sample(list(template_shape), yellow_count))

    all_shapes = [template_shape, *test_shapes]
    gaps = [random.randint(*GEN_GAP_RANGE) for _ in range(len(all_shapes) - 1)]
    left_margin = random.randint(*GEN_MARGIN_RANGE)
    right_margin = random.randint(*GEN_MARGIN_RANGE)

    placements: list[tuple[int, int, tuple[tuple[int, int], ...]]] = []
    cursor = left_margin
    for idx, shape in enumerate(all_shapes):
        sh, sw = _shape_size(shape)
        if sh > band_height:
            return None
        top = _sample_top(band_height, sh)
        placements.append((top, cursor, shape))
        cursor += sw
        if idx + 1 < len(all_shapes):
            cursor += gaps[idx]
    band_width = cursor + right_margin

    inp = np.full((band_height, band_width), BLACK, dtype=int)
    out = np.full_like(inp, BLACK)

    template_top, template_left, _template = placements[0]
    _render_template(inp, template_shape, template_top, template_left, yellow_cells)
    _render_template(out, template_shape, template_top, template_left, yellow_cells)

    for idx, (is_match, shape, color) in enumerate(zip(flags, test_shapes, colors), start=1):
        top, left, _ = placements[idx]
        _render_shape(inp, shape, top, left, color)
        _render_shape(out, shape, top, left, GREEN if is_match else RED)

    return inp, out


def _assemble_bands(bands: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    widths = [band_in.shape[1] for band_in, _band_out in bands]
    band_height = bands[0][0].shape[0]
    full_width = max(widths)
    full_height = len(bands) * band_height + (len(bands) - 1)

    inp = np.full((full_height, full_width), BLACK, dtype=int)
    out = np.full_like(inp, BLACK)

    row = 0
    for idx, (band_in, band_out) in enumerate(bands):
        h, w = band_in.shape
        inp[row : row + h, :w] = band_in
        out[row : row + h, :w] = band_out
        row += h
        if idx + 1 < len(bands):
            inp[row, :] = AZURE
            out[row, :] = AZURE
            row += 1

    return inp, out


def _generate_attempt() -> np.ndarray | None:
    band_count = random.randint(*GEN_BAND_COUNT_RANGE)
    band_height = random.randint(*GEN_BAND_HEIGHT_RANGE)

    bands: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(band_count):
        band = _build_band(band_height)
        if band is None:
            return None
        bands.append(band)

    inp, expected = _assemble_bands(bands)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
